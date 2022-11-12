use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let proto_files = &["npu_ir/dfg.proto", "npu_ir/common.proto"];
    let mut prost_config = prost_build::Config::new();
    let proto_includes = if let Ok(path) = std::env::var("NPU_TOOLS_PATH") {
        vec![PathBuf::from(path).join("crates/npu-ir/protos_generated/")]
    } else {
        vec![PathBuf::from("/usr/share/furiosa/protos")]
    };
    prost_config.type_attribute(".", "#[allow(clippy::large_enum_variant)]");
    prost_config.out_dir(&out_dir);
    prost_config.btree_map(&[".npu_ir"]);
    prost_config.compile_protos(proto_files, &proto_includes)?;

    Ok(())
}
