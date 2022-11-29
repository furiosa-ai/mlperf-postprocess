#![deny(unused_extern_crates)]
#![feature(vec_into_raw_parts)]
#![allow(clippy::borrow_deref_ref)]

use pyo3::prelude::*;

pub mod common;
pub mod resnet50;
pub mod ssd_large;
pub mod ssd_small;

#[pymodule]
fn furiosa_native_postprocess(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let py_sys_module = py.import("sys")?.getattr("modules")?;
    let ssd_mobilenet_module = pyo3::wrap_pymodule!(ssd_small::ssd_mobilenet);
    let ssd_resnet34_module = pyo3::wrap_pymodule!(ssd_large::ssd_resnet34);
    let resnet50_module = pyo3::wrap_pymodule!(resnet50::resnet50);

    m.add_wrapped(ssd_mobilenet_module)?;
    m.add_wrapped(ssd_resnet34_module)?;
    m.add_wrapped(resnet50_module)?;

    py_sys_module.set_item("furiosa_mlperf_postprocess.ssd_mobilenet", ssd_mobilenet_module(py))?;
    py_sys_module.set_item("furiosa_mlperf_postprocess.ssd_resnet34", ssd_resnet34_module(py))?;
    py_sys_module.set_item("furiosa_mlperf_postprocess.resnet50", resnet50_module(py))?;

    Ok(())
}
