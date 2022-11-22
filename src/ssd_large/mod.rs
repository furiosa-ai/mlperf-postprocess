#![allow(unused_extern_crates)]
extern crate openmp_sys;

use std::convert::TryInto;
use std::mem;

use itertools::Itertools;
use rayon::prelude::*;

use crate::common::graph::GraphInfo;
use crate::common::graph::TensorInfo;
use crate::common::ssd_postprocess::{
    BoundingBox, CenteredBox, DetectionResult, DetectionResults, Postprocess,
};
use crate::common::{
    shape::{LoweredShape, Shape},
    uninitialized_vec,
};

const FEATURE_MAP_SHAPES: [usize; 6] = [50, 25, 13, 7, 3, 3];
const NUM_ANCHORS: [usize; 6] = [4, 6, 6, 6, 4, 4];

// 50x50x4 + 25x25x6 + 13x13x6 + 7x7x6 + 3x3x4 + 3x3x4
const CHANNEL_COUNT: usize = 15130;
const NUM_CLASSES: usize = 81;
const SIZE_OF_F32: usize = mem::size_of::<f32>();
// 0~5 scores 6~11 boxes
const NUM_OUTPUTS: usize = 12;
const SCALE_XY: f32 = 0.1;
const SCALE_WH: f32 = 0.2;

const SCORE_THRESHOLD: f32 = 0.05f32;
const NMS_THRESHOLD: f32 = 0.5f32;
const MAX_DETECTION: usize = 200;

#[derive(Debug, Clone)]
pub struct RustPostprocessor {
    output_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_exp_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_exp_scale_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_base_index: [usize; 7],
    score_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2],
    box_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2],
    box_priors: Vec<CenteredBox>,
    parallel_processing: bool,
}

impl RustPostprocessor {
    pub fn new(main: &GraphInfo) -> Self {
        assert_eq!(main.outputs.len(), NUM_OUTPUTS);

        let mut output_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut output_exp_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut output_exp_scale_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut score_lowered_shapes = [Default::default(); NUM_OUTPUTS / 2];
        let mut box_lowered_shapes = [Default::default(); NUM_OUTPUTS / 2];
        for (i, tensor_index) in main.outputs.iter().enumerate() {
            let tensor: TensorInfo = main.tensors.get(tensor_index).unwrap().into();
            let (s, z) = tensor.get_scale_and_zero_point();
            let mut table = [0f32; 256];
            let mut exp_table = [0f32; 256];
            let mut exp_scale_table = [0f32; 256];
            for q in -128..=127 {
                let x = (s * f64::from(q - z)) as f32;
                let index = (q as u8) as usize;
                if i < 6 {
                    table[index] = x;
                } else {
                    table[index] = x * SCALE_XY;
                }
                exp_table[index] = f32::exp(x);
                exp_scale_table[index] = f32::exp(x * SCALE_WH);
            }
            output_deq_tables[i] = table;
            output_exp_deq_tables[i] = exp_table;
            output_exp_scale_deq_tables[i] = exp_scale_table;
            if let Some(i) = i.checked_sub(NUM_OUTPUTS / 2) {
                box_lowered_shapes[i] = tensor.get_lowered_shape();
            } else {
                score_lowered_shapes[i] = tensor.get_lowered_shape();
            }
        }

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

        Self {
            output_deq_tables,
            output_exp_deq_tables,
            output_exp_scale_deq_tables,
            output_base_index,
            score_lowered_shapes,
            box_lowered_shapes,
            box_priors,
            parallel_processing: false,
        }
    }

    #[must_use]
    pub fn with_parallel_processing(mut self, x: bool) -> Self {
        self.parallel_processing = x;
        self
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
        let mut results = if self.parallel_processing {
            let mut results = vec![Vec::new(); NUM_CLASSES - 1];
            results.par_iter_mut().enumerate().for_each(|(i, results)| {
                self.filter_result(query_index, scores, scores_sum, boxes, i + 1, results)
            });
            results.into_iter().flatten().collect_vec()
        } else {
            (1..NUM_CLASSES).fold(
                Vec::with_capacity(NUM_CLASSES * MAX_DETECTION),
                |mut results, i| {
                    self.filter_result(query_index, scores, scores_sum, boxes, i, &mut results);
                    results
                },
            )
        };

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(MAX_DETECTION);
        results.into()
    }

    fn decode_score_inner(&self, buffers: &[&[u8]], class_index: usize, scores: &mut [f32]) {
        for output_index in 0..6 {
            let shape = &self.score_lowered_shapes[output_index];
            let original_shape =
                Shape::new(FEATURE_MAP_SHAPES[output_index], FEATURE_MAP_SHAPES[output_index]);
            let num_anchor = NUM_ANCHORS[output_index];
            for anchor_index in 0..num_anchor {
                for h in 0..FEATURE_MAP_SHAPES[output_index] {
                    for w in 0..FEATURE_MAP_SHAPES[output_index] {
                        let c = class_index * num_anchor + anchor_index;
                        let q = buffers[output_index][shape.index(c, h, w)];
                        let score = self.output_exp_deq_tables[output_index][q as usize];
                        let scores_sum_index = original_shape.index(anchor_index, h, w)
                            + self.output_base_index[output_index];
                        scores[scores_sum_index] = score;
                    }
                }
            }
        }
    }

    fn decode_score(&self, buffers: &[&[u8]]) -> Vec<f32> {
        let mut scores = unsafe { uninitialized_vec(NUM_CLASSES * CHANNEL_COUNT) };

        if self.parallel_processing {
            scores.par_chunks_mut(CHANNEL_COUNT).enumerate().for_each(|(class_index, scores)| {
                self.decode_score_inner(buffers, class_index, scores);
            });
        } else {
            scores.chunks_mut(CHANNEL_COUNT).enumerate().for_each(|(class_index, scores)| {
                self.decode_score_inner(buffers, class_index, scores);
            });
        }
        scores
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

    fn decode_box(&self, boxes: &[&[u8]]) -> Vec<BoundingBox> {
        let mut ret = unsafe { uninitialized_vec(CHANNEL_COUNT) };

        let output_base_index = &self.output_base_index;
        let output_deq_tables = &self.output_deq_tables;
        let output_exp_scale_deq_tables = &self.output_exp_scale_deq_tables;
        let box_lowered_shapes = &self.box_lowered_shapes;
        for index in 0..6 {
            let deq_table: [f32; 256] = output_deq_tables[index + 6];
            let exp_scale_table: [f32; 256] = output_exp_scale_deq_tables[index + 6];
            let original_shape = Shape::new(FEATURE_MAP_SHAPES[index], FEATURE_MAP_SHAPES[index]);
            let shape = &box_lowered_shapes[index];
            let b = boxes[index];
            for f_y in 0..FEATURE_MAP_SHAPES[index] {
                for anchor_index in 0..NUM_ANCHORS[index] {
                    for f_x in 0..FEATURE_MAP_SHAPES[index] {
                        let q0 = b[shape.index(anchor_index, f_y, f_x)];
                        let q1 = b[shape.index(anchor_index + NUM_ANCHORS[index], f_y, f_x)];
                        let q2 = b[shape.index(anchor_index + 2 * NUM_ANCHORS[index], f_y, f_x)];
                        let q3 = b[shape.index(anchor_index + 3 * NUM_ANCHORS[index], f_y, f_x)];

                        let bx = CenteredBox {
                            pcy: deq_table[q1 as usize],
                            pcx: deq_table[q0 as usize],
                            ph: exp_scale_table[q3 as usize],
                            pw: exp_scale_table[q2 as usize],
                        };

                        let box_index =
                            original_shape.index(anchor_index, f_y, f_x) + output_base_index[index];

                        let prior_index =
                            output_base_index[index] + original_shape.index(anchor_index, f_y, f_x);
                        ret[box_index] = self.box_priors[prior_index].adjust(bx).into();
                    }
                }
            }
        }
        ret
    }
}

impl Postprocess for RustPostprocessor {
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "PostProcess", cat = "Mlperf"),
        skip(self, data)
    )]
    fn postprocess(&self, id: f32, data: &[&[u8]]) -> DetectionResults {
        let (scores, boxes) = data.split_at(6);
        let boxes = self.decode_box(boxes);
        debug_assert_eq!(boxes.len(), CHANNEL_COUNT);

        let scores = self.decode_score(scores);
        debug_assert_eq!(scores.len(), CHANNEL_COUNT * NUM_CLASSES); // 1,225,530

        let scores_sum = self.calculate_score_sum(&scores);
        debug_assert_eq!(scores_sum.len(), CHANNEL_COUNT);

        self.filter_results(id, &scores, &scores_sum, &boxes)
    }
}

#[cfg(feature = "cpp_impl")]
pub mod cxx {
    use cpp::cpp;

    use super::*;

    cpp! {{
        #include "cpp/unlower.h"
        #include "cpp/ssd_large.h"
        #include "bindings.h"
    }}

    #[derive(Debug, Clone)]
    pub struct CppPostprocessor;

    impl Postprocess for CppPostprocessor {
        #[allow(clippy::transmute_num_to_bytes)]
        fn postprocess(&self, index: f32, data: &[&[u8]]) -> DetectionResults {
            let data_ptr = data.as_ptr();

            let mut ret = Vec::with_capacity(200);
            let result_ptr = ret.as_mut_ptr();

            let n = cpp!(unsafe [index as "float", data_ptr as "const U8Slice*", result_ptr as "DetectionResult*"] -> usize as "size_t" {
                return ssd_large::post_inference<true>(index, data_ptr, result_ptr);
            });

            debug_assert!(n <= 200);
            unsafe {
                ret.set_len(n);
            }
            ret.into()
        }
    }

    impl CppPostprocessor {
        pub fn new(main: &GraphInfo) -> Self {
            assert_eq!(main.outputs.len(), NUM_OUTPUTS);

            let mut output_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
            let mut output_exp_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
            let mut output_exp_scale_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
            let mut score_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2] =
                [Default::default(); NUM_OUTPUTS / 2];
            let mut box_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2] =
                [Default::default(); NUM_OUTPUTS / 2];

            for (i, tensor_index) in main.outputs.iter().enumerate() {
                let tensor: TensorInfo = main.tensors.get(tensor_index).unwrap().into();
                let (s, z) = tensor.get_scale_and_zero_point();
                let mut table = [0f32; 256];
                let mut exp_table = [0f32; 256];
                let mut exp_scale_table = [0f32; 256];
                for q in -128..=127 {
                    let index = (q as u8) as usize;
                    let x = (s * f64::from(q - z)) as f32;
                    if i < 6 {
                        table[index] = x;
                    } else {
                        table[index] = x * SCALE_XY;
                    };
                    exp_table[index] = f32::exp(x);
                    exp_scale_table[index] = f32::exp(x * SCALE_WH);
                }
                if let Some(i) = i.checked_sub(NUM_OUTPUTS / 2) {
                    box_lowered_shapes[i] = tensor.get_lowered_shape();
                } else {
                    score_lowered_shapes[i] = tensor.get_lowered_shape();
                }

                output_deq_tables[i] = table;
                output_exp_deq_tables[i] = exp_table;
                output_exp_scale_deq_tables[i] = exp_scale_table;
            }

            let mut output_base_index = [0usize; 7];
            for i in 0..6 {
                output_base_index[i + 1] = output_base_index[i]
                    + NUM_ANCHORS[i] * FEATURE_MAP_SHAPES[i] * FEATURE_MAP_SHAPES[i];
            }

            {
                let score_lowered_shapes_ptr = score_lowered_shapes.as_ptr();
                let box_lowered_shapes_ptr = box_lowered_shapes.as_ptr();
                let output_deq_tables_ptr = output_deq_tables.as_ptr();
                let output_exp_deq_tables_ptr = output_exp_deq_tables.as_ptr();
                let output_exp_scale_deq_tables_ptr = output_exp_scale_deq_tables.as_ptr();
                let box_priors: Vec<f32> =
                    include_bytes!("../../models/ssd_large_precomputed_priors")
                        .chunks(SIZE_OF_F32 * 4)
                        .flat_map(|bytes| {
                            let (pcy, pcx, ph, pw) = bytes
                                .chunks(SIZE_OF_F32)
                                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                .tuples()
                                .next()
                                .unwrap();
                            CenteredBox { pcy, pcx, ph, pw }.into_transposed().to_vec()
                        })
                        .collect();

                let box_priors_ptr = box_priors.as_ptr();

                cpp!(unsafe [output_deq_tables_ptr as "float*", output_exp_deq_tables_ptr as "float*", output_exp_scale_deq_tables_ptr as "float*", score_lowered_shapes_ptr as "LoweredShapeFromRust*", box_lowered_shapes_ptr as "LoweredShapeFromRust*", box_priors_ptr as "CenteredBox*"] {
                    ssd_large::init(output_deq_tables_ptr, output_exp_deq_tables_ptr, output_exp_scale_deq_tables_ptr, score_lowered_shapes_ptr, box_lowered_shapes_ptr, box_priors_ptr);

                });
            }

            Self
        }
    }
}
