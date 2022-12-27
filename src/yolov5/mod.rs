use std::fmt;

use itertools::izip;
use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray, PyArray2, PyReadonlyArray3, PyReadonlyArray5};
use pyo3::prelude::*;

use crate::common::ssd_postprocess::{BoundingBox, CenteredBox};

/// Returns max value's index and value (Argmax, Max)
///
/// Copied from https://docs.rs/rulinalg/latest/src/rulinalg/utils.rs.html#245-261
pub fn argmax<T>(u: &[T]) -> (usize, T)
where
    T: Copy + PartialOrd,
{
    // Length is always nonzero
    // assert!(u.len() != 0);

    let mut max_index = 0;
    let mut max = u[max_index];

    for (i, v) in u.iter().enumerate().skip(1) {
        if max < *v {
            max_index = i;
            max = *v;
        }
    }

    (max_index, max)
}

#[derive(Debug, Clone)]
pub struct RustPostprocessor {
    pub num_classes: usize,
    pub num_outputs: usize,
    pub num_detection_layers: usize,
    pub num_anchor: usize,
    pub anchors: Array3<f32>,
    pub strides: Vec<f32>,
}

impl fmt::Display for RustPostprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RustPostProcessor {{ num_classes: {}, num_outputs: {}, \
            num_detection_layers: {}, num_anchor: {}, strides: {:?} }}",
            self.num_classes,
            self.num_outputs,
            self.num_detection_layers,
            self.num_anchor,
            self.strides
        )
    }
}

impl RustPostprocessor {
    fn new(anchors: Array3<f32>, num_classes: usize, strides: Vec<f32>) -> Self {
        let shape = anchors.shape();
        Self {
            num_classes,
            num_outputs: num_classes + 5,
            num_detection_layers: shape[0],
            num_anchor: shape[1],
            anchors,
            strides,
        }
    }

    fn box_decode(
        &self,
        inputs: Vec<PyReadonlyArray5<'_, f32>>,
        conf_threshold: f32,
    ) -> Array2<f32> {
        const MAX_BOXES: usize = 10_000;
        const NUM_COLS: usize = 6;
        let mut results = Vec::new();
        let mut num_rows: usize = 0;

        'outer: for (&stride, anchors_inner_stride, inner_stride) in
            izip!(&self.strides, self.anchors.outer_iter(), inputs)
        {
            for inner_batch in inner_stride.as_array().outer_iter() {
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
                            if object_confidence < conf_threshold {
                                continue;
                            };
                            let (max_class_idx, max_class_confidence) = argmax(class_confs);
                            // Low class confidence, skip
                            if object_confidence * max_class_confidence < conf_threshold {
                                continue;
                            }

                            // (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                            // (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                            // yolov5 boundingbox format(center_x,center_y,width,height)
                            let centered_box = CenteredBox::new_centered_box(
                                (by * 2.0 - 0.5 + y as f32) * stride,
                                (bx * 2.0 - 0.5 + x as f32) * stride,
                                4.0 * bh * bh * ay,
                                4.0 * bw * bw * ax,
                            );

                            // xywh -> xyxy
                            let bbox: BoundingBox = centered_box.into();

                            results.extend_from_slice(&[
                                bbox.px1,
                                bbox.py1,
                                bbox.px2,
                                bbox.py2,
                                max_class_confidence,
                                max_class_idx as f32,
                            ]);
                            num_rows += 1;
                            if num_rows >= MAX_BOXES {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        Array2::from_shape_vec((num_rows, NUM_COLS), results).unwrap()
    }
}

/// YOLOv5 PostProcessor
///
/// It takes anchors, class_names, strides as input
///
/// Args:
///     anchors (numpy.ndarray)
///     num_classes (int)
///     strides (numpy.ndarray)
#[pyclass]
#[pyo3(
    text_signature = "(anchors: numpy.ndarray, class_names: Sequence[str], strides: numpy.ndarray)"
)]
pub struct RustPostProcessor(RustPostprocessor);

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new(
        anchors: PyReadonlyArray3<'_, f32>,
        num_classes: usize,
        strides: Vec<f32>,
    ) -> PyResult<Self> {
        Ok(Self(RustPostprocessor::new(anchors.to_owned_array(), num_classes, strides)))
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
    ///     conf_threshold (float): confidence threshold
    ///
    /// Returns:
    ///     numpy.ndarray: Output tensors
    /// #[pyo3(text_signature = "(self, inputs: Sequence[numpy.ndarray], conf_threshold: float)")]
    fn eval(
        &self,
        py: Python<'_>,
        inputs: Vec<PyReadonlyArray5<'_, f32>>,
        conf_threshold: f32,
    ) -> PyResult<Py<PyArray2<f32>>> {
        Ok(PyArray::from_array(py, &self.0.box_decode(inputs, conf_threshold)).to_owned())
    }
}

#[pymodule]
pub(crate) fn yolov5(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}
