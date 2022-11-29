pub mod graph;
pub mod model;
pub mod proto;
pub mod shape;
pub mod ssd_postprocess;

use numpy::PyArrayDyn;
use pyo3::{self, exceptions::PyValueError, pyclass, pymethods, types::PyList, PyErr, PyResult};
use ssd_postprocess::DetectionResult;

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
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self)
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

pub(crate) fn convert_to_slices(inputs: &PyList) -> PyResult<Vec<&[u8]>> {
    let input_len = inputs.len();

    let mut memories: Vec<&[u8]> = unsafe { uninitialized_vec(input_len) };
    for (index, tensor) in inputs.into_iter().enumerate() {
        let tensor = tensor.downcast::<PyArrayDyn<i8>>()?;
        if !tensor.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "{}th tensor is not C-contiguous".to_string(),
            ));
        }
        let slice: &[u8] = unsafe {
            let raw_slice = tensor.as_slice()?;
            std::slice::from_raw_parts(raw_slice.as_ptr() as *const u8, raw_slice.len())
        };
        memories[index] = slice;
    }

    Ok(memories)
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
