pub mod utils;
use std::fmt;

use itertools::{izip, Itertools};
use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray2, PyReadonlyArray3, PyReadonlyArray5};
use pyo3::prelude::*;
use rayon::prelude::*;
use utils::{centered_box_to_ltrb_bulk, DetectionBoxes};

#[derive(Debug, Clone)]
pub struct RustPostprocessor {
    pub anchors: Array3<f32>,
    pub strides: Vec<f32>,
    pub agnostic: bool,
}

impl fmt::Display for RustPostprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.anchors.shape();
        write!(
            f,
            "RustPostProcessor {{ num_detection_layers: {}, num_anchor: {}, strides: {:?}, agnostic: {} }}",
            shape[0], shape[1], self.strides, self.agnostic
        )
    }
}

impl RustPostprocessor {
    fn new(anchors: Array3<f32>, strides: Vec<f32>, agnostic: Option<bool>) -> Self {
        pub const NUM_ANCHOR_LAST: usize = 2;
        assert_eq!(
            anchors.shape()[2],
            NUM_ANCHOR_LAST,
            "anchors' last dimension must be {NUM_ANCHOR_LAST}"
        );
        Self { anchors, strides, agnostic: agnostic.unwrap_or(false) }
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

        for (&stride, anchors_inner_stride, inner_stride) in
            izip!(&self.strides, self.anchors.outer_iter(), inputs)
        {
            for (batch_index, inner_batch) in inner_stride.as_array().outer_iter().enumerate() {
                // Perform box_decode for one batch
                let mut pcy: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut pcx: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut ph: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut pw: Vec<f32> = Vec::with_capacity(MAX_BOXES);

                let mut scores: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                let mut classes: Vec<f32> = Vec::with_capacity(MAX_BOXES);
                'outer: for (anchors, inner_anchor) in
                    izip!(anchors_inner_stride.outer_iter(), inner_batch.outer_iter())
                {
                    let &[ax, ay] = (anchors.to_owned() * stride).as_slice().unwrap() else {
                        unreachable!()
                    };
                    for (y, inner_y) in inner_anchor.outer_iter().enumerate() {
                        for (x, inner_x) in inner_y.outer_iter().enumerate() {
                            // Destruct output array
                            let &[bx, by, bw, bh, object_confidence, ref class_confs @ ..]: &[f32] =
                                inner_x.as_slice().expect("inner_x must be contiguous")
                            else {
                                unreachable!()
                            };

                            // Find candidates where `class_confidence * object_confidence > conf_threshold`
                            let candidates = class_confs
                                .iter()
                                .enumerate() // enumerate to store class index for later
                                .filter(|(_, &class_conf)| {
                                    class_conf * object_confidence > conf_threshold
                                })
                                .collect_vec();
                            if candidates.is_empty() {
                                continue;
                            }

                            // Decode box
                            // (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                            // (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                            let cy = (by * 2.0 - 0.5 + y as f32) * stride;
                            let cx = (bx * 2.0 - 0.5 + x as f32) * stride;
                            let h = 4.0 * bh * bh * ay;
                            let w = 4.0 * bw * bw * ax;

                            for (class_idx, class_conf) in candidates {
                                num_rows += 1;
                                if num_rows >= MAX_BOXES {
                                    break 'outer;
                                }

                                pcy.push(cy);
                                pcx.push(cx);
                                ph.push(h);
                                pw.push(w);
                                scores.push(class_conf * object_confidence);
                                classes.push(class_idx as f32);
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
    fn nms(
        boxes: &DetectionBoxes,
        iou_threshold: f32,
        epsilon: Option<f32>,
        agnostic: bool,
    ) -> Vec<usize> {
        const MAX_NMS: usize = 300;
        const MAX_WH: f32 = 7680.;
        let epsilon = epsilon.unwrap_or(1e-5);

        let c =
            if agnostic { Array1::zeros(boxes.len()) } else { boxes.classes.to_owned() * MAX_WH };
        let x1 = &boxes.x1 + &c;
        let y1 = &boxes.y1 + &c;
        let x2 = &boxes.x2 + &c;
        let y2 = &boxes.y2 + &c;

        let mut indices: Vec<usize> = (0..boxes.len()).collect();
        let mut results: Vec<usize> = Vec::new();

        let dx = (&x2 - &x1).map(|&v| f32::max(0., v));
        let dy = (&y2 - &y1).map(|&v| f32::max(0., v));
        let areas: Array1<f32> = dx * dy;

        // Performs unstable argmax `indices = argmax(boxes.scores)`
        indices.sort_unstable_by(|&i, &j| {
            let box_score_i = unsafe { boxes.scores.uget(i) };
            let box_score_j = unsafe { boxes.scores.uget(j) };
            box_score_i.partial_cmp(box_score_j).unwrap()
        });

        while let Some(cur_idx) = indices.pop() {
            if results.len() >= MAX_NMS {
                break;
            }
            results.push(cur_idx);

            let xx1: Array1<f32> = unsafe {
                indices.iter().map(|&i| f32::max(*x1.uget(cur_idx), *x1.uget(i))).collect()
            };
            let yy1: Array1<f32> = unsafe {
                indices.iter().map(|&i| f32::max(*y1.uget(cur_idx), *y1.uget(i))).collect()
            };
            let xx2: Array1<f32> = unsafe {
                indices.iter().map(|&i| f32::min(*x2.uget(cur_idx), *x2.uget(i))).collect()
            };
            let yy2: Array1<f32> = unsafe {
                indices.iter().map(|&i| f32::min(*y2.uget(cur_idx), *y2.uget(i))).collect()
            };

            let widths = (xx2 - xx1).mapv(|v| f32::max(0.0, v));
            let heights = (yy2 - yy1).mapv(|v| f32::max(0.0, v));

            let ious = widths * heights;
            let cut_areas: Array1<f32> =
                indices.iter().map(|&i| unsafe { *areas.uget(i) }).collect();
            let overlap = &ious / (unsafe { *areas.uget(cur_idx) } + cut_areas - &ious + epsilon);

            indices = indices
                .into_iter()
                .enumerate()
                .filter_map(|(i, j)| (unsafe { *overlap.uget(i) } <= iou_threshold).then_some(j))
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
        epsilon: Option<f32>,
        agnostic: Option<bool>,
    ) -> Vec<Array2<f32>> {
        const MAX_NMS_INPUT: usize = 30_000;
        let agnostic: bool = agnostic.unwrap_or(self.agnostic);

        self.box_decode(inputs, conf_threshold)
            .into_par_iter()
            .map(|mut dbox| {
                if dbox.len() > MAX_NMS_INPUT {
                    dbox.sort_by_score_and_trim(MAX_NMS_INPUT);
                };
                let indices = Self::nms(&dbox, iou_threshold, epsilon, agnostic);
                dbox.select_and_convert(&indices)
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
///     agnostic (Optional[bool]): Whether to use agnostic NMS, default is False
#[pyclass]
pub struct RustPostProcessor(RustPostprocessor);

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new(
        anchors: PyReadonlyArray3<'_, f32>,
        strides: Vec<f32>,
        agnostic: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self(RustPostprocessor::new(anchors.to_owned_array(), strides, agnostic)))
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
    ///     epsilon (Optional[float]): Epsilon for numerical stability
    ///     agnostic (Optional[bool]): Whether to use agnostic NMS, takes precedence constructor's
    ///
    /// Returns:
    ///     List[numpy.ndarray]: Batched detection results
    fn eval(
        &self,
        py: Python<'_>,
        inputs: Vec<PyReadonlyArray5<'_, f32>>,
        conf_threshold: f32,
        iou_threshold: f32,
        epsilon: Option<f32>,
        agnostic: Option<bool>,
    ) -> PyResult<Vec<Py<PyArray2<f32>>>> {
        Ok(self
            .0
            .postprocess(inputs, conf_threshold, iou_threshold, epsilon, agnostic)
            .into_iter()
            .map(|results| PyArray2::from_owned_array(py, results).to_owned())
            .collect())
    }
}

pub(crate) fn yolo(m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}
