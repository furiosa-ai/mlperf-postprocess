#![deny(unused_extern_crates)]
#![feature(vec_into_raw_parts)]
#![allow(clippy::borrow_deref_ref)]

use pyo3::prelude::*;

pub mod common;
pub mod ssd_large;
pub mod ssd_small;
pub mod yolo;

fn add_submodule(
    m: &PyModule,
    init_submodule: fn(&PyModule) -> PyResult<()>,
    name: &str,
) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), name)?;
    init_submodule(submodule)?;
    m.add_submodule(submodule)?;

    // Add module into Python modules dict as PyO3 could not recognize parent as package
    // See https://github.com/PyO3/pyo3/issues/759
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(format!("furiosa_native_postprocess.{name}"), submodule)?;

    Ok(())
}

#[pymodule]
fn furiosa_native_postprocess(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    const VERSION: &str = env!("CARGO_PKG_VERSION");

    m.add("__version__", VERSION)?;

    add_submodule(m, ssd_large::ssd_resnet34, "ssd_resnet34")?;
    add_submodule(m, ssd_small::ssd_mobilenet, "ssd_mobilenet")?;
    add_submodule(m, yolo::yolo, "yolo")?;

    // backward compatibility
    add_submodule(m, yolo::yolo, "yolov5")?;

    Ok(())
}
