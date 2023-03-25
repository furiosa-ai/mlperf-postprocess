#![allow(unused_extern_crates)]
extern crate openmp_sys;

use std::mem;

use itertools::Itertools;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use rayon::prelude::*;

use crate::common::graph::{create_graph_from_binary_with_header, GraphInfo};
use crate::common::model::ModelOutputInfo;
use crate::common::ssd_postprocess::{
    BoundingBox, CenteredBox, DetectionResult, DetectionResults, Postprocess,
};
use crate::common::{
    convert_to_slices,
    shape::{Shape, TensorIndexer},
    uninitialized_vec, PyDetectionResult,
};

const FEATURE_MAP_SHAPES: [usize; 6] = [19, 10, 5, 3, 2, 1];
const ANCHOR_STRIDES: [usize; 6] = [19 * 19, 10 * 10, 5 * 5, 3 * 3, 2 * 2, 1];
const NUM_ANCHORS: [usize; 6] = [3, 6, 6, 6, 6, 6];

// 19x19x3 + 10x10x6 + 5x5x6 + 3x3x6 + 2x2x6 + 1x1x6
const CHANNEL_COUNT: usize = 1917;
const NUM_CLASSES: usize = 91;
const SIZE_OF_F32: usize = mem::size_of::<f32>();
// 0~5 scores 6~11 boxes
const NUM_OUTPUTS: usize = 12;
const SCALE_XY: f32 = 0.1;
const SCALE_WH: f32 = 0.2;
const SCORE_THRESHOLD: f32 = 0.3f32;
const NMS_THRESHOLD: f32 = 0.6f32;

#[derive(Debug, Clone)]
pub struct RustPostprocessor {
    output_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_exp_scale_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_base_index: [usize; 7],
    score_thresholds: [Option<i8>; NUM_OUTPUTS / 2],
    score_lowered_shapes: [TensorIndexer; NUM_OUTPUTS / 2],
    box_lowered_shapes: [TensorIndexer; NUM_OUTPUTS / 2],
    box_priors: Vec<CenteredBox>,
    parallel_processing: bool,
}

impl RustPostprocessor {
    pub fn new(main: &GraphInfo) -> Self {
        let model: ModelOutputInfo = main.into();
        Self::from(&model)
    }

    #[must_use]
    pub fn with_parallel_processing(mut self, x: bool) -> Self {
        self.parallel_processing = x;
        self
    }

    fn filter_result(
        &self,
        query_index: f32,
        scores: &[&[u8]],
        boxes: &[BoundingBox],
        class_index: usize,
        results: &mut Vec<DetectionResult>,
        class_offset: usize,
    ) {
        let mut filtered = Vec::with_capacity(CHANNEL_COUNT);
        for index in 0..6 {
            if let Some(score_threshold) = self.score_thresholds[index] {
                let shape = &self.score_lowered_shapes[index];
                let original_shape =
                    Shape::new(FEATURE_MAP_SHAPES[index], FEATURE_MAP_SHAPES[index]);
                let deq_table = self.output_deq_tables[index];
                for anchor_index in 0..NUM_ANCHORS[index] {
                    for f_y in 0..FEATURE_MAP_SHAPES[index] {
                        for f_x in 0..FEATURE_MAP_SHAPES[index] {
                            let score_index =
                                shape.index(anchor_index * NUM_CLASSES + class_index, f_y, f_x);
                            let q = scores[index][score_index] as i8;
                            if q >= score_threshold {
                                let score = deq_table[q as u8 as usize];
                                debug_assert!(score > SCORE_THRESHOLD);
                                filtered.push((
                                    score,
                                    original_shape.index(anchor_index, f_y, f_x)
                                        + self.output_base_index[index],
                                ));
                            }
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
        scores: &[&[u8]],
        boxes: &[BoundingBox],
    ) -> DetectionResults {
        if self.parallel_processing {
            let mut results = vec![Vec::new(); NUM_CLASSES - 1];
            results.par_iter_mut().enumerate().for_each(|(i, results)| {
                self.filter_result(query_index, scores, boxes, i + 1, results, 0)
            });
            results.into_iter().flatten().collect_vec().into()
        } else {
            let mut results = Vec::with_capacity(CHANNEL_COUNT);
            for class_index in 1..NUM_CLASSES {
                let offset = results.len();
                self.filter_result(query_index, scores, boxes, class_index, &mut results, offset);
            }
            results.into()
        }
    }

    fn decode_box(&self, boxes: &[&[u8]]) -> Vec<BoundingBox> {
        let mut ret = unsafe { uninitialized_vec(CHANNEL_COUNT) };
        let output_deq_tables = &self.output_deq_tables;
        let output_exp_scale_deq_tables = &self.output_exp_scale_deq_tables;
        let output_base_index = &self.output_base_index;
        let box_lowered_shapes = &self.box_lowered_shapes;

        for index in 0..boxes.len() {
            let anchor_stride = ANCHOR_STRIDES[index];
            let deq_table = &output_deq_tables[index + 6];
            let exp_scale_table = &output_exp_scale_deq_tables[index + 6];
            let shape = &box_lowered_shapes[index];
            let b = boxes[index];
            debug_assert!(!b.is_empty());
            for anchor_index in 0..NUM_ANCHORS[index] {
                for f_y in 0..FEATURE_MAP_SHAPES[index] {
                    for f_x in 0..FEATURE_MAP_SHAPES[index] {
                        let feature_index = f_y * FEATURE_MAP_SHAPES[index] + f_x;
                        // TODO `b` 를 cache-friendly 하게 접근하는 lowered shape 고민
                        let q0 = b[shape.index(anchor_index * 4, f_y, f_x)];
                        let q1 = b[shape.index(anchor_index * 4 + 1, f_y, f_x)];
                        let q2 = b[shape.index(anchor_index * 4 + 2, f_y, f_x)];
                        let q3 = b[shape.index(anchor_index * 4 + 3, f_y, f_x)];

                        let bx = CenteredBox {
                            pcy: deq_table[q0 as usize],
                            pcx: deq_table[q1 as usize],
                            ph: exp_scale_table[q2 as usize],
                            pw: exp_scale_table[q3 as usize],
                        };

                        let box_index =
                            output_base_index[index] + feature_index + anchor_index * anchor_stride;
                        // TODO `prior_index` 대신 `box_index` 를 사용할 수 있도록 `self.box_priors` 의 배열 변경
                        let prior_index = output_base_index[index]
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

impl<'a> From<&'a ModelOutputInfo> for RustPostprocessor {
    fn from(model: &'a ModelOutputInfo) -> Self {
        assert_eq!(model.outputs.len(), NUM_OUTPUTS);

        let mut output_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut output_exp_scale_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut score_lowered_shapes = [Default::default(); NUM_OUTPUTS / 2];
        let mut box_lowered_shapes = [Default::default(); NUM_OUTPUTS / 2];
        let mut score_thresholds = [Default::default(); NUM_OUTPUTS / 2];
        for (i, tensor_meta) in model.outputs.iter().enumerate() {
            let (s, z) = tensor_meta.get_scale_and_zero_point();
            let mut table = [0f32; 256];
            let mut exp_scale_table = [0f32; 256];
            for q in -128..=127 {
                let index = (q as u8) as usize;
                let x = (s * f64::from(q - z)) as f32;
                if i < 6 {
                    table[index] = f32::exp(x) / (1f32 + f32::exp(x));
                } else {
                    table[index] = x * SCALE_XY;
                    exp_scale_table[index] = f32::exp(x * SCALE_WH);
                };
            }
            if let Some(i) = i.checked_sub(NUM_OUTPUTS / 2) {
                box_lowered_shapes[i] = tensor_meta.indexer;
            } else {
                score_lowered_shapes[i] = tensor_meta.indexer;

                score_thresholds[i] = (i8::MIN..=i8::MAX).find(|&q| {
                    let index = q as u8 as usize;
                    table[index] > SCORE_THRESHOLD
                });
            }

            output_deq_tables[i] = table;
            output_exp_scale_deq_tables[i] = exp_scale_table;
        }

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

        Self {
            output_deq_tables,
            output_exp_scale_deq_tables,
            output_base_index,
            score_thresholds,
            score_lowered_shapes,
            box_lowered_shapes,
            box_priors,
            parallel_processing: false,
        }
    }
}

impl Postprocess for RustPostprocessor {
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "PostProcess", cat = "Mlperf"),
        skip(self, data)
    )]
    fn postprocess(&self, index: f32, data: &[&[u8]]) -> DetectionResults {
        let (scores, boxes) = data.split_at(6);
        let boxes = self.decode_box(boxes);
        debug_assert_eq!(boxes.len(), CHANNEL_COUNT);
        self.filter_results(index, scores, &boxes)
    }
}

#[cfg(feature = "cpp_impl")]
pub mod cxx {
    use cpp::cpp;

    use super::*;

    cpp! {{
        #include "cpp/unlower.h"
        #include "cpp/ssd_small.h"
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
                return ssd_small::post_inference<true>(index, data_ptr, result_ptr);
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
            let model: ModelOutputInfo = main.into();
            Self::from(&model)
        }
    }

    impl<'a> From<&'a ModelOutputInfo> for CppPostprocessor {
        fn from(model: &'a ModelOutputInfo) -> Self {
            assert_eq!(model.outputs.len(), NUM_OUTPUTS);

            let mut output_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
            let mut output_exp_scale_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
            let mut score_lowered_shapes: [TensorIndexer; NUM_OUTPUTS / 2] =
                [Default::default(); NUM_OUTPUTS / 2];
            let mut box_lowered_shapes: [TensorIndexer; NUM_OUTPUTS / 2] =
                [Default::default(); NUM_OUTPUTS / 2];

            for (i, tensor_meta) in model.outputs.iter().enumerate() {
                let (s, z) = tensor_meta.get_scale_and_zero_point();
                let mut table = [0f32; 256];
                let mut exp_scale_table = [0f32; 256];
                for q in -128..=127 {
                    let index = (q as u8) as usize;
                    let x = (s * f64::from(q - z)) as f32;
                    if i < 6 {
                        table[index] = f32::exp(x) / (1f32 + f32::exp(x));
                    } else {
                        table[index] = x * SCALE_XY;
                        exp_scale_table[index] = f32::exp(x * SCALE_WH);
                    };
                }
                if let Some(i) = i.checked_sub(NUM_OUTPUTS / 2) {
                    box_lowered_shapes[i] = tensor_meta.indexer;
                } else {
                    score_lowered_shapes[i] = tensor_meta.indexer;
                }

                output_deq_tables[i] = table;
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
                let output_exp_scale_deq_tables_ptr = output_exp_scale_deq_tables.as_ptr();

                let box_priors: Vec<CenteredBox> =
                    include_bytes!("../../models/ssd_small_precomputed_priors")
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
                let box_priors_ptr = box_priors.as_ptr();

                cpp!(unsafe [output_deq_tables_ptr as "float*", output_exp_scale_deq_tables_ptr as "float*", score_lowered_shapes_ptr as "LoweredShapeFromRust*", box_lowered_shapes_ptr as "LoweredShapeFromRust*", box_priors_ptr as "CenteredBox*"] {
                    ssd_small::init(output_deq_tables_ptr, output_exp_scale_deq_tables_ptr, score_lowered_shapes_ptr, box_lowered_shapes_ptr, box_priors_ptr);

                });
            }

            Self
        }
    }

    #[pymethods]
    impl CppPostProcessor {
        #[new]
        fn new(dfg: &[u8]) -> PyResult<Self> {
            let graph = create_graph_from_binary_with_header(dfg)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid DFG format: {}", e)))?;

            Ok(Self(CppPostprocessor::new(&graph)))
        }

        /// Evaluate the postprocess
        ///
        /// Args:
        ///     inputs (Sequence[numpy.ndarray]): Input tensors
        ///
        /// Returns:
        ///     List[PyDetectionResult]: Output tensors
        #[pyo3(text_signature = "(self, inputs: Sequence[numpy.ndarray])")]
        fn eval(&self, inputs: &PyList) -> PyResult<Vec<PyDetectionResult>> {
            if inputs.len() != OUTPUT_NUM {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "expected {} input tensors but got {}",
                    OUTPUT_NUM,
                    inputs.len()
                )));
            }

            let slices = convert_to_slices(inputs)?;
            Ok(self
                .0
                .postprocess(0f32, &slices)
                .0
                .into_iter()
                .map(PyDetectionResult::new)
                .collect())
        }
    }

    /// CppPostProcessor
    ///
    /// It takes a DFG whose unlower part is removed.
    /// The DFG binary must have magic number in its head.
    ///
    /// Args:
    ///     dfg (bytes): a binary of DFG IR
    #[pyclass]
    #[pyo3(text_signature = "(dfg: bytes)")]
    pub struct CppPostProcessor(CppPostprocessor);
}

const OUTPUT_NUM: usize = 12;

/// RustPostProcessor
///
/// It takes a DFG whose unlower part is removed.
/// The DFG binary must have magic number in its head.
///
/// Args:
///     dfg (bytes): a binary of DFG IR
#[pyclass]
#[pyo3(text_signature = "(dfg: bytes)")]
pub struct RustPostProcessor(RustPostprocessor);

#[pymethods]
impl RustPostProcessor {
    #[new]
    fn new(dfg: &[u8]) -> PyResult<Self> {
        let graph = create_graph_from_binary_with_header(dfg)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid DFG format: {e}")))?;

        Ok(Self(RustPostprocessor::new(&graph)))
    }

    /// Evaluate the postprocess
    ///
    /// Args:
    ///     inputs (Sequence[numpy.ndarray]): Input tensors
    ///
    /// Returns:
    ///     List[PyDetectionResult]: Output tensors
    #[pyo3(text_signature = "(self, inputs: Sequence[numpy.ndarray])")]
    fn eval(&self, inputs: &PyList) -> PyResult<Vec<PyDetectionResult>> {
        if inputs.len() != OUTPUT_NUM {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected {} input tensors but got {}",
                OUTPUT_NUM,
                inputs.len()
            )));
        }

        let slices = convert_to_slices(inputs)?;
        Ok(self.0.postprocess(0f32, &slices).0.into_iter().map(PyDetectionResult::new).collect())
    }
}

#[pymodule]
pub(crate) fn ssd_mobilenet(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPostProcessor>()?;

    #[cfg(feature = "cpp_impl")]
    {
        m.add_class::<cxx::CppPostProcessor>()?;
    }

    Ok(())
}
