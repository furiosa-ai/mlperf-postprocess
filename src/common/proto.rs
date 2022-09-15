pub mod common {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.common.rs"));
}

pub mod dfg {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.dfg.rs"));
}
