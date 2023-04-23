use std::collections::HashMap;

use crc::{self, Crc};
use eyre::ensure;
use prost::Message;

use super::model::{ModelOutputInfo, TensorMeta};
use crate::common::model::{ElementType, QuantizationParameter};
use crate::common::proto;
use crate::common::shape::TensorIndexer;

pub type TensorIndex = u32;

impl<'a> From<&'a proto::element_type::ElementType> for ElementType {
    fn from(element_type: &'a proto::element_type::ElementType) -> Self {
        match element_type.inner.as_ref().unwrap() {
            proto::element_type::element_type::Inner::Int8(inner) => {
                let quantization_info = inner.quantization_info.as_ref().unwrap();
                assert_eq!(quantization_info.quantization_parameters.len(), 1);

                let proto::element_type::QuantizationParameter { min, max } =
                    quantization_info.quantization_parameters[0];

                let scale = (max - min) / (i8::max_value() as f64 - i8::min_value() as f64);
                let zero_point = i8::min_value() as i32 - (min / scale).round() as i32;
                Self::Int8 { quantization_parameter: QuantizationParameter { scale, zero_point } }
            }
            proto::element_type::element_type::Inner::Uint8(inner) => {
                let quantization_info = inner.quantization_info.as_ref().unwrap();
                assert_eq!(quantization_info.quantization_parameters.len(), 1);

                let proto::element_type::QuantizationParameter { min, max } =
                    quantization_info.quantization_parameters[0];

                let scale = (max - min) / (u8::max_value() as f64 - u8::min_value() as f64);
                let zero_point = u8::min_value() as i32 - (min / scale).round() as i32;
                Self::Uint8 { quantization_parameter: QuantizationParameter { scale, zero_point } }
            }
            _ => unimplemented!("Only Int8/Uint8 output type supported"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    shape: TensorIndexer,
    element_type: ElementType,
}

impl<'a> From<&'a proto::tensor::Tensor> for TensorInfo {
    fn from(tensor: &'a proto::tensor::Tensor) -> Self {
        Self {
            shape: tensor.shape.as_ref().unwrap().into(),
            element_type: tensor.element_type.as_ref().unwrap().into(),
        }
    }
}

impl TensorInfo {
    pub fn get_scale_and_zero_point(&self) -> (f64, i32) {
        self.element_type.get_scale_and_zero_point()
    }

    pub fn get_lowered_shape(&self) -> TensorIndexer {
        self.shape
    }
}

#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub outputs: Vec<TensorIndex>,
    pub tensors: HashMap<TensorIndex, proto::tensor::Tensor>,
}

#[allow(clippy::from_over_into)]
impl<'a> Into<ModelOutputInfo> for &'a GraphInfo {
    fn into(self) -> ModelOutputInfo {
        let mut outputs = Vec::new();

        for output in self.outputs.iter() {
            let tensor_info: TensorInfo = self
                .tensors
                .get(output)
                .expect("Output tensor should exist in the tensor map")
                .into();
            outputs.push(TensorMeta {
                indexer: tensor_info.shape,
                element_type: tensor_info.element_type,
            })
        }

        ModelOutputInfo { outputs }
    }
}

pub fn create_graph_from_binary(input: &[u8]) -> eyre::Result<GraphInfo> {
    let graph = proto::dfg::Graph::decode(input)?;

    let tensors = graph
        .tensors
        .iter()
        .map(|(&tensor_index, tensor)| (tensor_index, tensor.clone()))
        .collect::<HashMap<TensorIndex, proto::tensor::Tensor>>();

    Ok(GraphInfo { outputs: graph.outputs, tensors })
}

pub fn create_graph_from_binary_with_header(input: &[u8]) -> eyre::Result<GraphInfo> {
    let (header, body) = BinaryHeader::detach_header(input)?;

    assert_eq!(header.magic_number, BinaryHeader::create_magic_number(EXTENSION_DFG));
    ensure!(header.crc == BinaryHeader::create_crc(body), "Error: cyclic redundancy check failed",);

    create_graph_from_binary(body)
}

// Belows are copied from npu-tools

pub const EXTENSION_DFG: &str = "dfg";

pub const MAGIC_NUMBER_LENGTH: usize = 3;
pub const CRC_LENGTH: usize = 4;
pub const VERSION_LENGTH: usize = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Version {
    major: u8,
    minor: u8,
    patch: u8,
}

impl TryFrom<Vec<u8>> for Version {
    type Error = eyre::Report;

    fn try_from(vec: Vec<u8>) -> Result<Self, Self::Error> {
        ensure!(
            vec.len() == VERSION_LENGTH,
            "the length of version should be {:?}), but {:?}.",
            VERSION_LENGTH,
            vec.len(),
        );
        Ok(Self { major: vec[0], minor: vec[1], patch: vec[2] })
    }
}

struct BinaryHeader {
    pub magic_number: Vec<u8>,
    pub crc: Vec<u8>,
    #[allow(dead_code)]
    pub version: Version,
}

impl BinaryHeader {
    fn detach_header(buf: &[u8]) -> eyre::Result<(Self, &[u8])> {
        ensure!(
            buf.len() >= MAGIC_NUMBER_LENGTH + CRC_LENGTH + VERSION_LENGTH,
            "Bytecode should have a binary header"
        );
        let (magic_number, remainder) = buf.split_at(MAGIC_NUMBER_LENGTH);
        let (crc, remainder) = remainder.split_at(CRC_LENGTH);
        let (version, body) = remainder.split_at(VERSION_LENGTH);
        Ok((
            Self {
                magic_number: magic_number.into(),
                crc: crc.into(),
                version: version.to_vec().try_into()?,
            },
            body,
        ))
    }

    fn create_magic_number(buf: &str) -> Vec<u8> {
        let mut bytes = buf.to_string().into_bytes();
        bytes.resize_with(MAGIC_NUMBER_LENGTH, Default::default);
        bytes
    }

    fn create_crc(buf: &[u8]) -> Vec<u8> {
        // See https://github.com/mrhooray/crc-rs/issues/62 for CRC32_ISO_HDLC.
        const CRC32: Crc<u32> = Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
        let crc = CRC32.checksum(buf);
        crc.to_le_bytes().to_vec()
    }
}
