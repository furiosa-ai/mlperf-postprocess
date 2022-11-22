use crate::common::shape::TensorIndexer;

#[derive(Debug, Clone)]
pub struct QuantizationParameter {
    pub scale: f64,
    pub zero_point: i32,
}

#[derive(Debug, Clone)]
pub enum ElementType {
    Float64,
    Int8 { quantization_parameter: QuantizationParameter },
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub indexer: TensorIndexer,
    pub element_type: ElementType,
}

#[derive(Debug, Clone)]
pub struct ModelOutputInfo {
    pub outputs: Vec<TensorMeta>,
}
