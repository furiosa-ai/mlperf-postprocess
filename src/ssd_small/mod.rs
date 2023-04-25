#![allow(unused_extern_crates)]
extern crate openmp_sys;

use std::mem;

use itertools::Itertools;
use ndarray::Array3;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use rayon::prelude::*;

use crate::common::ssd_postprocess::{BoundingBox, CenteredBox, DetectionResult, DetectionResults};
use crate::common::{downcast_to_f32, uninitialized_vec, PyDetectionResult};

const FEATURE_MAP_SHAPES: [usize; 6] = [19, 10, 5, 3, 2, 1];
const ANCHOR_STRIDES: [usize; 6] = [19 * 19, 10 * 10, 5 * 5, 3 * 3, 2 * 2, 1];
const NUM_ANCHORS: [usize; 6] = [3, 6, 6, 6, 6, 6];

// 19x19x3 + 10x10x6 + 5x5x6 + 3x3x6 + 2x2x6 + 1x1x6
const CHANNEL_COUNT: usize = 1917;
const NUM_CLASSES: usize = 91;
const SIZE_OF_F32: usize = mem::size_of::<f32>();
const SCALE_XY: f32 = 0.1;
const SCALE_WH: f32 = 0.2;
const SCORE_THRESHOLD: f32 = 0.3f32;
const NMS_THRESHOLD: f32 = 0.6f32;

#[derive(Debug, Clone)]
pub struct RustPostprocessor {
    output_base_index: [usize; 7],
    box_priors: Vec<CenteredBox>,
}

impl Default for RustPostprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl RustPostprocessor {
    pub fn new() -> Self {
        let mut output_base_index = [0usize; 7];
        for i in 0..6 {
            output_base_index[i + 1] = output_base_index[i]
                + NUM_ANCHORS[i] * FEATURE_MAP_SHAPES[i] * FEATURE_MAP_SHAPES[i];
        }

        let box_priors = include_bytes!("../../models/ssd_small_precomputed_priors")
            .chunks(SIZE_OF_F32 * 4)
            .map(|bytes| {
                let (py1, px1, py2, px2) = bytes
                    .chunks(SIZE_OF_F32)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .tuples()
                    .next()
                    .unwrap();
                BoundingBox { py1, px1, py2, px2 }.into()
            })
            .collect();

        Self { output_base_index, box_priors }
    }

    fn filter_result(
        &self,
        query_index: f32,
        scores: &[Array3<f32>],
        boxes: &[BoundingBox],
        class_index: usize,
        results: &mut Vec<DetectionResult>,
        class_offset: usize,
    ) {
        let mut filtered = Vec::with_capacity(CHANNEL_COUNT);
        for index in 0..6 {
            for anchor_index in 0..NUM_ANCHORS[index] {
                for f_y in 0..FEATURE_MAP_SHAPES[index] {
                    for f_x in 0..FEATURE_MAP_SHAPES[index] {
                        let q = scores[index]
                            .get((anchor_index * NUM_CLASSES + class_index, f_y, f_x))
                            .unwrap();
                        if *q >= SCORE_THRESHOLD {
                            filtered.push((
                                *q,
                                self.output_base_index[index]
                                    + f_y * FEATURE_MAP_SHAPES[index]
                                    + f_x
                                    + anchor_index * ANCHOR_STRIDES[index],
                            ));
                        }
                    }
                }
            }
        }

        filtered.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for (score, box_index) in filtered {
            let candidate = &boxes[box_index];
            if results[class_offset..].iter().all(|r| candidate.iou(&r.bbox) <= NMS_THRESHOLD) {
                results.push(DetectionResult {
                    index: query_index,
                    bbox: *candidate,
                    score,
                    class: class_index as f32,
                });
            }
        }
    }

    fn filter_results(
        &self,
        query_index: f32,
        scores: &[Array3<f32>],
        boxes: &[BoundingBox],
    ) -> DetectionResults {
        let mut results = vec![Vec::new(); NUM_CLASSES - 1];
        results.par_iter_mut().enumerate().for_each(|(i, results)| {
            self.filter_result(query_index, scores, boxes, i + 1, results, 0)
        });
        results.into_iter().flatten().collect_vec().into()
    }

    fn decode_box(&self, boxes: &[Array3<f32>]) -> Vec<BoundingBox> {
        let mut ret = unsafe { uninitialized_vec(CHANNEL_COUNT) };

        for (index, b) in boxes.iter().enumerate() {
            let anchor_stride = ANCHOR_STRIDES[index];

            debug_assert!(!b.is_empty());
            for anchor_index in 0..NUM_ANCHORS[index] {
                for f_y in 0..FEATURE_MAP_SHAPES[index] {
                    for f_x in 0..FEATURE_MAP_SHAPES[index] {
                        let feature_index = f_y * FEATURE_MAP_SHAPES[index] + f_x;

                        let q0 = *b.get((anchor_index * 4, f_y, f_x)).unwrap();
                        let q1 = *b.get((anchor_index * 4 + 1, f_y, f_x)).unwrap();
                        let q2 = *b.get((anchor_index * 4 + 2, f_y, f_x)).unwrap();
                        let q3 = *b.get((anchor_index * 4 + 3, f_y, f_x)).unwrap();

                        let q2 = f32::exp(q2 * SCALE_WH / SCALE_XY);
                        let q3 = f32::exp(q3 * SCALE_WH / SCALE_XY);

                        let bx = CenteredBox { pcy: q0, pcx: q1, ph: q2, pw: q3 };

                        let box_index = self.output_base_index[index]
                            + feature_index
                            + anchor_index * anchor_stride;
                        // TODO `prior_index` 대신 `box_index` 를 사용할 수 있도록 `self.box_priors` 의 배열 변경
                        let prior_index = self.output_base_index[index]
                            + feature_index * NUM_ANCHORS[index]
                            + anchor_index;

                        ret[box_index] = self.box_priors[prior_index].adjust(bx).into();
                    }
                }
            }
        }
        ret
    }
}

impl RustPostprocessor {
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "PostProcess", cat = "Mlperf"),
        skip(self, scores, boxes)
    )]
    fn postprocess(
        &self,
        index: f32,
        scores: &[Array3<f32>],
        boxes: &[Array3<f32>],
    ) -> DetectionResults {
        let boxes = self.decode_box(boxes);
        debug_assert_eq!(boxes.len(), CHANNEL_COUNT);
        self.filter_results(index, scores, &boxes)
    }
}

const BOXES_NUM: usize = 6;
const SCORES_NUM: usize = 6;

/// RustPostProcessor
///
/// It takes a DFG whose unlower part is removed.
/// The DFG binary must have magic number in its head.
///
/// Args:
///     dfg (bytes): a binary of DFG IR
// FIXME: Rename the struct. We can customize the python class name (see https://docs.rs/pyo3/latest/pyo3/attr.pyclass.html)
#[pyclass]
pub struct RustPostProcessor(RustPostprocessor);

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self(RustPostprocessor::new()))
    }

    /// Evaluate the postprocess
    ///
    /// Args:
    ///     inputs (Sequence[numpy.ndarray]): Input tensors
    ///
    /// Returns:
    ///     List[PyDetectionResult]: Output tensors
    fn eval(&self, boxes: &PyList, scores: &PyList) -> PyResult<Vec<PyDetectionResult>> {
        if boxes.len() != BOXES_NUM {
            return Err(PyValueError::new_err(format!(
                "expected {BOXES_NUM} input boxes but got {}",
                boxes.len()
            )));
        }
        if scores.len() != SCORES_NUM {
            return Err(PyValueError::new_err(format!(
                "expected {SCORES_NUM} input scores but got {}",
                scores.len()
            )));
        }

        // TODO: improve conversion below

        let boxes = downcast_to_f32(boxes)?;
        let scores = downcast_to_f32(scores)?;

        let boxes2 = boxes.iter().map(|a| a.readonly()).collect_vec();
        let scores2 = scores.iter().map(|a| a.readonly()).collect_vec();

        let boxes3 = boxes2.iter().map(|a| a.as_array()).collect_vec();
        let scores3 = scores2.iter().map(|a| a.as_array()).collect_vec();

        let boxes4 = boxes3
            .iter()
            .map(|b| ndarray::Zip::from(b).map_collect(|t| t * SCALE_XY))
            .collect_vec();
        let scores4 = scores3
            .iter()
            .map(|b| ndarray::Zip::from(b).map_collect(|&t| f32::exp(t) / (1f32 + f32::exp(t))))
            .collect_vec();

        // TODO: assert shape here

        Ok(self
            .0
            .postprocess(0f32, &scores4, &boxes4)
            .0
            .into_iter()
            .map(PyDetectionResult::new)
            .collect())
    }
}

#[pymodule]
pub(crate) fn ssd_mobilenet(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}
