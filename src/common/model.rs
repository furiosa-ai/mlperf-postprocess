use crate::common::shape::TensorIndexer;

#[derive(Debug, Clone, Copy)]
pub struct QuantizationParameter {
    pub scale: f64,
    pub zero_point: i32,
}

#[derive(Debug, Clone)]
pub enum ElementType {
    Float64,
    Int8 { quantization_parameter: QuantizationParameter },
}

impl ElementType {
    pub fn get_scale_and_zero_point(&self) -> (f64, i32) {
        let ElementType::Int8 { quantization_parameter} = self else {
            todo!("Currently only supports Int8");
        };
        (quantization_parameter.scale, quantization_parameter.zero_point)
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub indexer: TensorIndexer,
    pub element_type: ElementType,
}

impl TensorMeta {
    pub fn get_scale_and_zero_point(&self) -> (f64, i32) {
        self.element_type.get_scale_and_zero_point()
    }
}

#[derive(Debug, Clone)]
pub struct ModelOutputInfo {
    pub outputs: Vec<TensorMeta>,
}
