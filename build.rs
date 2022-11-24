use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    #[cfg(feature = "cpp_impl")]
    {
        cbindgen::Builder::new()
            .with_crate(env::var("CARGO_MANIFEST_DIR").unwrap())
            .with_config(cbindgen::Config {
                include_guard: Some("bindings_h".to_string()),
                language: cbindgen::Language::Cxx,
                ..cbindgen::Config::default()
            })
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(out_dir.join("bindings.h"));

        cpp_build::Config::new()
            .flag("-march=native")
            .flag("-std=c++17")
            .flag("-fopenmp")
            .include(&out_dir)
            .opt_level(3)
            .build("src/lib.rs");
    }

    let proto_files = &["npu_ir/dfg.proto", "npu_ir/common.proto"];
    let mut prost_config = prost_build::Config::new();
    let proto_includes = if let Ok(path) = std::env::var("NPU_TOOLS_PATH") {
        vec![PathBuf::from(path).join("crates/npu-ir/protos_generated/")]
    } else {
        vec![PathBuf::from("/usr/share/furiosa/protos")]
    };
    prost_config.type_attribute(".", "#[allow(clippy::large_enum_variant)]");
    prost_config.out_dir(&out_dir);
    prost_config.btree_map([".npu_ir"]);
    prost_config.compile_protos(proto_files, &proto_includes)?;

    Ok(())
}
