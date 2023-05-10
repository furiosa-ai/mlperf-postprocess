#![allow(unused_extern_crates)]
extern crate openmp_sys;

use std::convert::TryInto;
use std::mem;

use itertools::Itertools;
use ndarray::Array3;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use rayon::prelude::*;

use crate::common::ssd_postprocess::{BoundingBox, CenteredBox, DetectionResult, DetectionResults};
use crate::common::{downcast_to_f32, uninitialized_vec, PyDetectionResult};

const FEATURE_MAP_SHAPES: [usize; 6] = [50, 25, 13, 7, 3, 3];
const ANCHOR_STRIDES: [usize; 6] = [50 * 50, 25 * 25, 13 * 13, 7 * 7, 3 * 3, 3 * 3];
const NUM_ANCHORS: [usize; 6] = [4, 6, 6, 6, 4, 4];

// 50x50x4 + 25x25x6 + 13x13x6 + 7x7x6 + 3x3x4 + 3x3x4
const CHANNEL_COUNT: usize = 15130;
const NUM_CLASSES: usize = 81;
const SIZE_OF_F32: usize = mem::size_of::<f32>();
const SCALE_XY: f32 = 0.1;
const SCALE_WH: f32 = 0.2;

const SCORE_THRESHOLD: f32 = 0.05f32;
const NMS_THRESHOLD: f32 = 0.5f32;
const MAX_DETECTION: usize = 200;

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

        let box_priors = include_bytes!("../../models/ssd_large_precomputed_priors")
            .chunks(SIZE_OF_F32 * 4)
            .map(|bytes| {
                let (pcy, pcx, ph, pw) = bytes
                    .chunks(SIZE_OF_F32)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .tuples()
                    .next()
                    .unwrap();
                CenteredBox { pcy, pcx, ph, pw }.into_transposed()
            })
            .collect();

        Self { output_base_index, box_priors }
    }

    fn filter_result(
        &self,
        query_index: f32,
        scores: &[f32],
        scores_sum: &[f32],
        boxes: &[BoundingBox],
        class_index: usize,
        results: &mut Vec<DetectionResult>,
    ) {
        let mut filtered = Vec::with_capacity(CHANNEL_COUNT);

        for i in 0..CHANNEL_COUNT {
            let score = scores[class_index * CHANNEL_COUNT + i] / scores_sum[i];
            if score > SCORE_THRESHOLD {
                filtered.push((score, i));
            }
        }

        filtered.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        filtered.truncate(MAX_DETECTION);

        let class_offset = results.len();
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
        scores: &[f32],
        scores_sum: &[f32],
        boxes: &[BoundingBox],
    ) -> DetectionResults {
        let mut results = {
            let mut results = vec![Vec::new(); NUM_CLASSES - 1];
            results.par_iter_mut().enumerate().for_each(|(i, results)| {
                self.filter_result(query_index, scores, scores_sum, boxes, i + 1, results)
            });
            results.into_iter().flatten().collect_vec()
        };

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(MAX_DETECTION);
        results.into()
    }

    fn decode_score_inner(&self, scores: &[Array3<f32>], class_index: usize, decoded: &mut [f32]) {
        for output_index in 0..6 {
            let num_anchor = NUM_ANCHORS[output_index];
            for anchor_index in 0..num_anchor {
                for h in 0..FEATURE_MAP_SHAPES[output_index] {
                    for w in 0..FEATURE_MAP_SHAPES[output_index] {
                        let c = class_index * num_anchor + anchor_index;
                        let scores_sum_index = self.output_base_index[output_index]
                            + h * FEATURE_MAP_SHAPES[output_index]
                            + w
                            + anchor_index * ANCHOR_STRIDES[output_index];

                        decoded[scores_sum_index] = *scores[output_index].get((c, h, w)).unwrap();
                    }
                }
            }
        }
    }

    fn decode_score(&self, scores: &[Array3<f32>]) -> Vec<f32> {
        let mut ret = unsafe { uninitialized_vec(NUM_CLASSES * CHANNEL_COUNT) };

        ret.par_chunks_mut(CHANNEL_COUNT).enumerate().for_each(|(class_index, ret)| {
            self.decode_score_inner(scores, class_index, ret);
        });
        ret
    }

    fn calculate_score_sum(&self, scores: &[f32]) -> Vec<f32> {
        let mut scores_sum = vec![0f32; CHANNEL_COUNT];
        for class_index in 0..NUM_CLASSES {
            for i in 0..CHANNEL_COUNT {
                scores_sum[i] += scores[class_index * CHANNEL_COUNT + i];
            }
        }
        scores_sum
    }

    fn decode_box(&self, boxes: &[Array3<f32>]) -> Vec<BoundingBox> {
        let mut ret = unsafe { uninitialized_vec(CHANNEL_COUNT) };

        for (index, b) in boxes.iter().enumerate() {
            let anchor_stride = ANCHOR_STRIDES[index];

            for f_y in 0..FEATURE_MAP_SHAPES[index] {
                for anchor_index in 0..NUM_ANCHORS[index] {
                    for f_x in 0..FEATURE_MAP_SHAPES[index] {
                        let feature_index = f_y * FEATURE_MAP_SHAPES[index] + f_x;

                        let pcx = *b.get((anchor_index, f_y, f_x)).unwrap();
                        let pcy = *b.get((anchor_index + NUM_ANCHORS[index], f_y, f_x)).unwrap();
                        let unscaled_pw =
                            *b.get((anchor_index + 2 * NUM_ANCHORS[index], f_y, f_x)).unwrap();
                        let unscaled_ph =
                            *b.get((anchor_index + 3 * NUM_ANCHORS[index], f_y, f_x)).unwrap();

                        let pw = f32::exp(unscaled_pw * SCALE_WH / SCALE_XY);
                        let ph = f32::exp(unscaled_ph * SCALE_WH / SCALE_XY);

                        let bx = CenteredBox { pcy, pcx, ph, pw };

                        let box_index = self.output_base_index[index]
                            + feature_index
                            + anchor_index * anchor_stride;

                        ret[box_index] = self.box_priors[box_index].adjust(bx).into();
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
        id: f32,
        scores: &[Array3<f32>],
        boxes: &[Array3<f32>],
    ) -> DetectionResults {
        let boxes = self.decode_box(boxes);
        debug_assert_eq!(boxes.len(), CHANNEL_COUNT);

        let scores = self.decode_score(scores);
        debug_assert_eq!(scores.len(), CHANNEL_COUNT * NUM_CLASSES); // 1,225,530

        let scores_sum = self.calculate_score_sum(&scores);
        debug_assert_eq!(scores_sum.len(), CHANNEL_COUNT);

        self.filter_results(id, &scores, &scores_sum, &boxes)
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

        let boxes = downcast_to_f32(boxes)?;
        let scores = downcast_to_f32(scores)?;

        let mut scaled_boxes = vec![];
        let mut exp_scores = vec![];
        for b in boxes {
            scaled_boxes.push(ndarray::Zip::from(b.as_array()).map_collect(|t| t * SCALE_XY));
        }
        for s in scores {
            exp_scores.push(ndarray::Zip::from(s.as_array()).map_collect(|&t| f32::exp(t)));
        }

        Ok(self
            .0
            .postprocess(0f32, &exp_scores, &scaled_boxes)
            .0
            .into_iter()
            .map(PyDetectionResult::new)
            .collect())
    }
}

#[pymodule]
pub(crate) fn ssd_resnet34(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    Ok(())
}
