pub mod axis {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.axis.rs"));
}

pub mod buffer_type {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.buffer_type.rs"));
}

pub mod common {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.common.rs"));
}

pub mod element_type {
    #![allow(clippy::enum_variant_names, clippy::module_inception)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.element_type.rs"));
}

pub mod pass_ops {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.pass_ops.rs"));
}

pub mod shape {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.shape.rs"));
}

pub mod tensor {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.tensor.rs"));
}

pub(crate) mod tu_ops {
    #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.rs"));

    pub(crate) mod renegade {
        #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
        include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.renegade.rs"));
    }

    pub(crate) mod warboy {
        #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
        include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.rs"));

        pub(crate) mod commit_unit {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.commit_unit.rs"));
        }

        pub(crate) mod fetch_unit {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.fetch_unit.rs"));

            pub(crate) mod fetch_network {
                #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
                include!(concat!(
                    env!("OUT_DIR"),
                    "/npu_ir.tu_ops.warboy.fetch_unit.fetch_network.rs"
                ));
            }

            pub(crate) mod fetch_sequencer {
                #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
                include!(concat!(
                    env!("OUT_DIR"),
                    "/npu_ir.tu_ops.warboy.fetch_unit.fetch_sequencer.rs"
                ));
            }
        }

        pub(crate) mod global_adder_tree {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.global_adder_tree.rs"));
        }

        pub(crate) mod load_register_file {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.load_register_file.rs"));
        }

        pub(crate) mod load_tensor_register_file {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(
                env!("OUT_DIR"),
                "/npu_ir.tu_ops.warboy.load_tensor_register_file.rs"
            ));
        }

        pub(crate) mod load_vector_register_file {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(
                env!("OUT_DIR"),
                "/npu_ir.tu_ops.warboy.load_vector_register_file.rs"
            ));
        }

        pub(crate) mod operation_unit {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.operation_unit.rs"));

            pub(crate) mod dot_product_engine {
                #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
                include!(concat!(
                    env!("OUT_DIR"),
                    "/npu_ir.tu_ops.warboy.operation_unit.dot_product_engine.rs"
                ));
            }

            pub(crate) mod feed_buffer {
                #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
                include!(concat!(
                    env!("OUT_DIR"),
                    "/npu_ir.tu_ops.warboy.operation_unit.feed_buffer.rs"
                ));
            }

            pub(crate) mod tensor_register_file_sequencer {
                #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
                include!(concat!(
                    env!("OUT_DIR"),
                    "/npu_ir.tu_ops.warboy.operation_unit.tensor_register_file_sequencer.rs"
                ));
            }
        }

        pub(crate) mod transpose_engine {
            #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
            include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_ops.warboy.transpose_engine.rs"));
        }
    }
}

pub(crate) mod tu_contraction {
    #![allow(clippy::enum_variant_names, clippy::derive_partial_eq_without_eq)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.tu_contraction.rs"));
}

pub mod operator {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.operator.rs"));
}

pub mod dfg {
    #![allow(clippy::enum_variant_names)]
    include!(concat!(env!("OUT_DIR"), "/npu_ir.dfg.rs"));
}
