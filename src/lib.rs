#![deny(unused_extern_crates)]
#![feature(vec_into_raw_parts)]
#![allow(clippy::borrow_deref_ref)]

use pyo3::prelude::*;

pub mod common;
// pub mod ssd_large;
pub mod ssd_small;
pub mod yolov5;

#[pymodule]
fn furiosa_native_postprocess(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let py_sys_module: &pyo3::types::PyDict = py.import("sys")?.getattr("modules")?.downcast()?;
    let ssd_mobilenet_module = pyo3::wrap_pymodule!(ssd_small::ssd_mobilenet);
    // let ssd_resnet34_module = pyo3::wrap_pymodule!(ssd_large::ssd_resnet34);
    let yolov5_module = pyo3::wrap_pymodule!(yolov5::yolov5);

    m.add_wrapped(ssd_mobilenet_module)?;
    // m.add_wrapped(ssd_resnet34_module)?;
    m.add_wrapped(yolov5_module)?;

    py_sys_module
        .set_item("furiosa_native_postprocess.ssd_mobilenet", m.getattr("ssd_mobilenet")?)?;
    // py_sys_module
    //     .set_item("furiosa_native_postprocess.ssd_resnet34", m.getattr("ssd_resnet34")?)?;
    py_sys_module.set_item("furiosa_native_postprocess.yolov5", m.getattr("yolov5")?)?;

    Ok(())
}
