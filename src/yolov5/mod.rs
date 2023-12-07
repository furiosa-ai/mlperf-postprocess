pub mod utils;
use std::fmt;

use itertools::izip;
use ndarray::{Array1, Array3};
use numpy::{PyReadonlyArray3, PyReadonlyArray5};
use pyo3::prelude::*;
use utils::{argmax, centered_box_to_ltrb_bulk, DetectionBoxes};

use crate::common::ssd_postprocess::{BoundingBox, DetectionResult, DetectionResults};
use crate::common::PyDetectionResults;

#[derive(Debug, Clone)]
pub struct RustPostprocessor {
    pub anchors: Array3<f32>,
    pub strides: Vec<f32>,
}

impl fmt::Display for RustPostprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.anchors.shape();
        write!(
            f,
            "RustPostProcessor {{ num_detection_layers: {}, num_anchor: {}, strides: {:?} }}",
            shape[0], shape[1], self.strides
        )
    }
}

impl RustPostprocessor {
    fn new(anchors: Array3<f32>, strides: Vec<f32>) -> Self {
        pub const NUM_ANCHOR_LAST: usize = 2;
        assert_eq!(
            anchors.shape()[2],
            NUM_ANCHOR_LAST,
            "anchors' last dimension must be {NUM_ANCHOR_LAST}"
        );
        Self { anchors, strides }
    }

    fn box_decode(
        &self,
        inputs: Vec<PyReadonlyArray5<'_, f32>>,
        conf_threshold: f32,
    ) -> Vec<DetectionBoxes> {
        const MAX_BOXES: usize = 10_000;
        let mut num_rows: usize = 0;

        let batch_size = inputs[0].shape()[0];
        let mut detection_boxes: Vec<DetectionBoxes> = vec![DetectionBoxes::empty(); batch_size];

        'outer: for (&stride, anchors_inner_stride, inner_stride) in
            izip!(&self.strides, self.anchors.outer_iter(), inputs)
        {
            for (batch_index, inner_batch) in inner_stride.as_array().outer_iter().enumerate() {
                // Perform box_decode for one batch
                let mut pcy: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut pcx: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut ph: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut pw: Vec<f32> = Vec::with_capacity(MAX_BOXES);

                let mut scores: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut classes: Vec<usize> = Vec::with_capacity(MAX_BOXES);
                for (anchors, inner_anchor) in
                    izip!(anchors_inner_stride.outer_iter(), inner_batch.outer_iter())
                {
                    let &[ax, ay] = (anchors.to_owned() * stride).as_slice().unwrap() else {
                        unreachable!()
                    };
                    for (y, inner_y) in inner_anchor.outer_iter().enumerate() {
                        for (x, inner_x) in inner_y.outer_iter().enumerate() {
                            // Destruct output array
                            let &[bx, by, bw, bh, object_confidence, ref class_confs @ ..] =
                                inner_x.as_slice().unwrap()
                            else {
                                unreachable!()
                            };

                            // Low object confidence, skip
                            if object_confidence <= conf_threshold {
                                continue;
                            };
                            let (max_class_idx, max_class_confidence) = argmax(class_confs);
                            // Low class confidence, skip
                            if object_confidence * max_class_confidence <= conf_threshold {
                                continue;
                            }

                            // (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                            // (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                            // yolov5 boundingbox format(center_x,center_y,width,height)
                            pcy.push((by * 2.0 - 0.5 + y as f32) * stride);
                            pcx.push((bx * 2.0 - 0.5 + x as f32) * stride);
                            ph.push(4.0 * bh * bh * ay);
                            pw.push(4.0 * bw * bw * ax);

                            // scores.push(object_confidence * max_class_confidence);
                            scores.push(object_confidence);
                            classes.push(max_class_idx);

                            num_rows += 1;
                            if num_rows >= MAX_BOXES {
                                break 'outer;
                            }
                        }
                    }
                }
                // Convert centered boxes to LTRB boxes at once
                let (x1, y1, x2, y2): (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) =
                    centered_box_to_ltrb_bulk(&pcy.into(), &pcx.into(), &pw.into(), &ph.into());
                detection_boxes[batch_index].append(x1, y1, x2, y2, scores.into(), classes.into());
            }
        }

        detection_boxes
    }

    /// Non-Maximum Suppression Algorithm
    /// Faster implementation by Malisiewicz et al.
    fn nms(boxes: &DetectionBoxes, iou_threshold: f32, epsilon: Option<f32>) -> Vec<usize> {
        let epsilon = epsilon.unwrap_or(1e-5);
        let mut indices: Vec<usize> = (0..boxes.len).collect();
        let mut results: Vec<usize> = Vec::new();

        let dx = (&boxes.x2 - &boxes.x1).map(|&v| f32::max(0., v));
        let dy = (&boxes.y2 - &boxes.y1).map(|&v| f32::max(0., v));
        let areas: Array1<f32> = dx * dy;

        // Performs unstable argmax `indices = argmax(boxes.scores)`
        indices.sort_unstable_by(|&i, &j| boxes.scores[i].partial_cmp(&boxes.scores[j]).unwrap());

        while let Some(cur_idx) = indices.pop() {
            results.push(cur_idx);

            let xx1: Array1<f32> =
                indices.iter().map(|&i| f32::max(boxes.x1[cur_idx], boxes.x1[i])).collect();
            let yy1: Array1<f32> =
                indices.iter().map(|&i| f32::max(boxes.y1[cur_idx], boxes.y1[i])).collect();
            let xx2: Array1<f32> =
                indices.iter().map(|&i| f32::min(boxes.x2[cur_idx], boxes.x2[i])).collect();
            let yy2: Array1<f32> =
                indices.iter().map(|&i| f32::min(boxes.y2[cur_idx], boxes.y2[i])).collect();

            let widths = (xx2 - xx1).mapv(|v| f32::max(0.0, v));
            let heights = (yy2 - yy1).mapv(|v| f32::max(0.0, v));

            let ious = widths * heights;
            let cut_areas: Array1<f32> = indices.iter().map(|&i| areas[i]).collect();
            let overlap = &ious / (areas[cur_idx] + cut_areas - &ious + epsilon);
            indices = (0..indices.len())
                .filter(|&i| overlap[i] <= iou_threshold)
                .map(|i| indices[i])
                .collect();
        }

        results
    }

    /// YOLOv5 postprocess function
    /// The vector in function input/output is for batched input/output
    fn postprocess(
        &self,
        inputs: Vec<PyReadonlyArray5<'_, f32>>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Vec<DetectionResults> {
        let max_nms: usize = 30000;
        let mut detection_boxes = self.box_decode(inputs, conf_threshold);
        // Inner vector for the result indexes in one image, outer vector for batch
        let indices: Vec<Vec<usize>> = detection_boxes
            .iter_mut()
            .map(|dbox| {
                if dbox.len > max_nms {
                    dbox.sort_by_score_and_trim(max_nms);
                };
                Self::nms(dbox, iou_threshold, None)
            })
            .collect();

        izip!(detection_boxes, indices)
            .map(|(dbox, indexes)| {
                DetectionResults(
                    indexes
                        .into_iter()
                        .map(|i| {
                            DetectionResult::new_detection_result(
                                i as f32,
                                BoundingBox::new_bounding_box(
                                    dbox.y1[i], dbox.x1[i], dbox.y2[i], dbox.x2[i],
                                ),
                                dbox.scores[i],
                                dbox.classes[i] as f32,
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

/// YOLOv5 PostProcessor
///
/// It takes anchors, class_names, strides as input
///
/// Args:
///     anchors (numpy.ndarray): Anchors (3D Array)
///     strides (numpy.ndarray): Strides (1D Array)
#[pyclass]
pub struct RustPostProcessor(RustPostprocessor);

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new(anchors: PyReadonlyArray3<'_, f32>, strides: Vec<f32>) -> PyResult<Self> {
        Ok(Self(RustPostprocessor::new(anchors.to_owned_array(), strides)))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.0))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }

    /// Evaluate the postprocess
    ///
    /// Args:
    ///     inputs (Sequence[numpy.ndarray]): Input tensors
    ///     conf_threshold (float): Confidence threshold
    ///     iou_threshold (float): IoU threshold
    ///
    /// Returns:
    ///     List[numpy.ndarray]: Batched detection results
    fn eval(
        &self,
        inputs: Vec<PyReadonlyArray5<'_, f32>>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> PyResult<Vec<PyDetectionResults>> {
        Ok(self
            .0
            .postprocess(inputs, conf_threshold, iou_threshold)
            .into_iter()
            .map(PyDetectionResults::from)
            .collect())
    }
}

#[pymodule]
pub(crate) fn yolov5(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}
