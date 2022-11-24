use crate::common::graph::GraphInfo;
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
