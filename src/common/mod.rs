pub mod graph;
pub mod model;
pub mod proto;
pub mod shape;
pub mod ssd_postprocess;

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
