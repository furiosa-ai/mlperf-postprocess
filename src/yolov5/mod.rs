use std::fmt;

use itertools::{izip, Itertools};
use numpy::ndarray::{s, Array2, Array3, ArrayView5, Dim};
use numpy::{PyArray, PyReadonlyArray3, PyReadonlyArray5};
use pyo3::prelude::*;

use crate::common::ssd_postprocess::{BoundingBox, CenteredBox};

/// YOLOv5 PostProcessor
///
/// It takes anchors, class_names, strides as input
///
/// Args:
///     anchors (numpy.ndarray)
///     class_names (List[str])
///     strides (numpy.ndarray)
#[pyclass]
#[pyo3(
    text_signature = "(anchors: numpy.ndarray, class_names: Sequence[str], strides: numpy.ndarray)"
)]
#[derive(Debug, Clone)]
pub struct RustPostProcessor {
    #[pyo3(get)]
    pub num_classes: usize,
    #[pyo3(get)]
    pub num_outputs: usize,
    #[pyo3(get)]
    pub num_detection_layers: usize,
    #[pyo3(get)]
    pub num_anchor: usize,
    pub anchors: Array3<f32>,
    #[pyo3(get)]
    pub class_names: Vec<String>,
    #[pyo3(get)]
    pub strides: Vec<f32>,
}

impl fmt::Display for RustPostProcessor {
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

pub fn argmax<T>(u: &[T]) -> (usize, T)
where
    T: Copy + PartialOrd,
{
    // Copied from https://docs.rs/rulinalg/latest/src/rulinalg/utils.rs.html#245-261
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

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new(
        anchors: PyReadonlyArray3<'_, f32>,
        class_names: Vec<String>,
        strides: Vec<f32>,
    ) -> Self {
        let anchors = anchors.to_owned_array();
        let shape = anchors.shape();
        Self {
            num_classes: class_names.len(),
            num_outputs: class_names.len() + 5,
            num_detection_layers: shape[0],
            num_anchor: shape[1],
            anchors,
            class_names,
            strides,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
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
    ) -> PyResult<Py<PyArray<f32, Dim<[usize; 2]>>>> {
        const CONFIDENCE_IDX: usize = 4;
        const MAX_BOXES: usize = 10_000;
        const NUM_COLS: usize = 6;
        let mut results = Vec::new();
        let mut num_rows: usize = 0;

        for (anchors, &stride, input) in
            izip!(self.anchors.outer_iter(), &self.strides, inputs.iter())
        {
            let full_array: ArrayView5<'_, f32> = input.as_array();
            let (&bs, &na, &ny, &nx, &no) =
                full_array.shape().iter().collect_tuple().expect("Wrong array dimension");
            assert_eq!(no, self.num_outputs, "Invalid number of output size");
            assert_eq!(na, self.num_anchor, "Wrong anchor number");
            assert_eq!(nx, ny, "Input must be square");

            'outer: for batch in 0..bs {
                for anchor_idx in 0..na {
                    let ax = anchors[[anchor_idx, 0]] * stride;
                    let ay = anchors[[anchor_idx, 1]] * stride;
                    for y in 0..ny {
                        for x in 0..nx {
                            let object_confidence =
                                full_array[[batch, anchor_idx, y, x, CONFIDENCE_IDX]];
                            // Low object confidence, skip
                            if object_confidence < conf_threshold {
                                continue;
                            }

                            // Array is 1-dimensional number_output lengthed array
                            let arr = full_array
                                .slice(s![batch, anchor_idx, y, x, ..])
                                .to_slice()
                                .unwrap();

                            // Get class-wise max confidence & index
                            let &[bx, by, bw, bh, _, ref class_confs@..] = arr else { unreachable!() };
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
        Ok(PyArray::from_array(py, &Array2::from_shape_vec((num_rows, NUM_COLS), results).unwrap())
            .to_owned())
    }
}

#[pymodule]
pub(crate) fn yolov5(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}
