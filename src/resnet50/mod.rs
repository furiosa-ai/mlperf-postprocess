use crate::common::graph::GraphInfo;
use crate::common::model::ModelOutputInfo;
use crate::common::{graph::TensorInfo, shape::TensorIndexer};

#[derive(Default, Debug, Clone)]
pub struct Resnet50PostProcessor {
    lowered_output_shape: TensorIndexer,
}

impl Resnet50PostProcessor {
    pub fn new(graph: &GraphInfo) -> Self {
        assert_eq!(graph.outputs.len(), 1, "number of output tensors should be 1");

        let output_tensor: TensorInfo = graph.tensors.get(&graph.outputs[0]).unwrap().into();
        Self { lowered_output_shape: output_tensor.get_lowered_shape() }
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
