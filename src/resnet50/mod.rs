use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};

use crate::common::convert_to_slices;
use crate::common::graph::{create_graph_from_binary_with_header, GraphInfo};
use crate::common::model::ModelOutputInfo;
use crate::common::shape::TensorIndexer;

#[derive(Default, Debug, Clone)]
pub struct Resnet50PostProcessor {
    lowered_output_shape: TensorIndexer,
}

impl Resnet50PostProcessor {
    pub fn new(graph: &GraphInfo) -> Self {
        let model: ModelOutputInfo = graph.into();
        Self::from(&model)
    }

    pub fn postprocess(&self, output: &[u8]) -> usize {
        (0..1001).max_by_key(|&c| output[self.lowered_output_shape.index(c, 0, 0)] as i8).unwrap()
    }
}

impl<'a> From<&'a ModelOutputInfo> for Resnet50PostProcessor {
    fn from(model: &'a ModelOutputInfo) -> Self {
        assert_eq!(model.outputs.len(), 1, "number of output tensors should be 1");

        Self { lowered_output_shape: model.outputs.get(0).as_ref().unwrap().indexer }
    }
}

const OUTPUT_NUM: usize = 1;

/// PostProcessor
///
/// It takes a DFG whose unlower part is removed.
/// The DFG binary must have magic number in its head.
///
/// Args:
///     dfg (bytes): a binary of DFG IR
#[pyclass]
#[pyo3(text_signature = "(dfg: bytes)")]
pub struct PostProcessor(Resnet50PostProcessor);

#[pymethods]
impl PostProcessor {
    #[new]
    fn new(dfg: &[u8]) -> PyResult<Self> {
        let graph = create_graph_from_binary_with_header(dfg)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid DFG format: {e}")))?;

        Ok(Self(Resnet50PostProcessor::new(&graph)))
    }

    /// Evaluate the postprocess
    ///
    /// Args:
    ///     inputs (Sequence[numpy.ndarray]): Input tensors
    ///
    /// Returns:
    ///     List[PyDetectionResult]: Output tensors
    #[pyo3(text_signature = "(self, inputs: Sequence[numpy.ndarray])")]
    fn eval(&self, inputs: &PyList) -> PyResult<usize> {
        if inputs.len() != OUTPUT_NUM {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected {} input tensors but got {}",
                OUTPUT_NUM,
                inputs.len()
            )));
        }

        let slices = convert_to_slices(inputs)?;
        Ok(self.0.postprocess(slices[0]))
    }
}

#[pymodule]
pub(crate) fn resnet50(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PostProcessor>()?;

    Ok(())
}
