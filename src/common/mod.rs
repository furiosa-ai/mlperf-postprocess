pub mod ssd_postprocess;

use numpy::PyArray3;
use pyo3::{self, pyclass, pymethods, types::PyList, PyResult};
use ssd_postprocess::{DetectionResult, DetectionResults};

#[pyclass]
#[derive(Clone, Debug)]
pub struct BoundingBox {
    #[pyo3(get)]
    left: f32,
    #[pyo3(get)]
    top: f32,
    #[pyo3(get)]
    right: f32,
    #[pyo3(get)]
    bottom: f32,
}

#[pymethods]
impl BoundingBox {
    fn __repr__(&self) -> String {
        format!(
            "BoundingBox(left: {}, top: {}, right: {}, bottom: {})",
            self.left, self.top, self.right, self.bottom
        )
    }

    fn __str__(&self) -> String {
        format!(
            "(left: {}, top: {}, right: {}, bottom: {})",
            self.left, self.top, self.right, self.bottom
        )
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyDetectionResult {
    #[pyo3(get)]
    pub left: f32,
    #[pyo3(get)]
    pub right: f32,
    #[pyo3(get)]
    pub top: f32,
    #[pyo3(get)]
    pub bottom: f32,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub class_id: i32,
}

#[pymethods]
impl PyDetectionResult {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:?}")
    }
}

impl PyDetectionResult {
    pub fn new(r: DetectionResult) -> Self {
        PyDetectionResult {
            left: r.bbox.px1,
            right: r.bbox.px2,
            top: r.bbox.py1,
            bottom: r.bbox.py2,
            score: r.score,
            class_id: r.class as i32,
        }
    }
}

pub type PyDetectionResults = Vec<PyDetectionResult>;

impl From<DetectionResults> for PyDetectionResults {
    fn from(value: DetectionResults) -> Self {
        value.0.into_iter().map(PyDetectionResult::new).collect()
    }
}

pub(crate) fn downcast_to_f32(inputs: &PyList) -> PyResult<Vec<&PyArray3<f32>>> {
    let mut ret = Vec::with_capacity(inputs.len());

    for tensor in inputs.into_iter() {
        let tensor = tensor.downcast::<PyArray3<f32>>()?;

        ret.push(tensor);
    }

    Ok(ret)
}

// u8slice
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct U8Slice {
    pub ptr: *const u8,
    pub len: usize,
}

impl U8Slice {
    #[no_mangle]
    pub extern "C" fn new_u8_slice(ptr: *const u8, len: usize) -> U8Slice {
        U8Slice { ptr, len }
    }
}

// uninitialized_vec
#[inline]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let (ptr, _, capacity) = Vec::with_capacity(size).into_raw_parts();
    Vec::from_raw_parts(ptr, size, capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unittest_slice_layout() {
        let v = vec![9u8; 1000];
        let slice = v.as_slice();
        let U8Slice { ptr, len } = unsafe { std::mem::transmute::<_, U8Slice>(slice) };
        assert_eq!(ptr, slice.as_ptr());
        assert_eq!(len, slice.len());
    }
}
