use std::ops::{Deref, DerefMut};
use std::{mem, slice};

pub trait Postprocess {
    fn postprocess(&self, index: f32, data: &[&[u8]]) -> DetectionResults;
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BoundingBox {
    pub py1: f32,
    pub px1: f32,
    pub py2: f32,
    pub px2: f32,
}

impl From<CenteredBox> for BoundingBox {
    fn from(cbox: CenteredBox) -> Self {
        Self { py1: cbox.py1(), px1: cbox.px1(), py2: cbox.py2(), px2: cbox.px2() }
    }
}

impl BoundingBox {
    #[no_mangle]
    pub extern "C" fn new_bounding_box(py1: f32, px1: f32, py2: f32, px2: f32) -> BoundingBox {
        BoundingBox { py1, px1, py2, px2 }
    }

    #[inline]
    pub fn pw(&self) -> f32 {
        self.px2 - self.px1
    }

    #[inline]
    pub fn ph(&self) -> f32 {
        self.py2 - self.py1
    }

    #[inline]
    pub fn pcx(&self) -> f32 {
        self.px1 + self.pw() * 0.5
    }

    #[inline]
    pub fn pcy(&self) -> f32 {
        self.py1 + self.ph() * 0.5
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.pw() * self.ph()
    }

    #[inline]
    pub fn transpose(&mut self) {
        mem::swap(&mut self.px1, &mut self.py1);
        mem::swap(&mut self.px2, &mut self.py2);
    }

    #[inline]
    #[must_use]
    pub fn into_transposed(mut self) -> Self {
        self.transpose();
        self
    }

    #[inline]
    pub fn iou(&self, other: &Self) -> f32 {
        let clamp_lower = |x: f32| {
            if x < 0f32 {
                0f32
            } else {
                x
            }
        };
        let cw = clamp_lower(f32::min(self.px2, other.px2) - f32::max(self.px1, other.px1));
        let ch = clamp_lower(f32::min(self.py2, other.py2) - f32::max(self.py1, other.py1));
        let overlap = cw * ch;
        overlap / (self.area() + other.area() - overlap + 0.00001f32)
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CenteredBox {
    pub pcy: f32,
    pub pcx: f32,
    pub ph: f32,
    pub pw: f32,
}

impl From<BoundingBox> for CenteredBox {
    fn from(prior: BoundingBox) -> Self {
        Self { pw: prior.pw(), ph: prior.ph(), pcx: prior.pcx(), pcy: prior.pcy() }
    }
}

impl CenteredBox {
    #[no_mangle]
    pub extern "C" fn new_centered_box(pcy: f32, pcx: f32, ph: f32, pw: f32) -> CenteredBox {
        CenteredBox { pcy, pcx, ph, pw }
    }

    pub fn to_vec(self) -> Vec<f32> {
        vec![self.pcy, self.pcx, self.ph, self.pw]
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn px1(&self) -> f32 {
        self.pcx - self.pw * 0.5
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn px2(&self) -> f32 {
        self.pcx + self.pw * 0.5
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn py1(&self) -> f32 {
        self.pcy - self.ph * 0.5
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn py2(&self) -> f32 {
        self.pcy + self.ph * 0.5
    }

    #[inline]
    pub fn transpose(&mut self) {
        mem::swap(&mut self.pcx, &mut self.pcy);
        mem::swap(&mut self.pw, &mut self.ph);
    }

    #[inline]
    #[must_use]
    pub fn into_transposed(mut self) -> Self {
        self.transpose();
        self
    }

    #[must_use]
    #[inline]
    #[no_mangle]
    pub extern "C" fn adjust(&self, x: Self) -> Self {
        Self {
            pcx: self.pcx + x.pcx * self.pw,
            pcy: self.pcy + x.pcy * self.ph,
            pw: x.pw * self.pw,
            ph: x.ph * self.ph,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DetectionResult {
    pub index: f32,
    pub bbox: BoundingBox,
    pub score: f32,
    pub class: f32,
}

impl DetectionResult {
    #[no_mangle]
    pub extern "C" fn new_detection_result(
        index: f32,
        bbox: BoundingBox,
        score: f32,
        class: f32,
    ) -> DetectionResult {
        DetectionResult { index, bbox, score, class }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct DetectionResults(pub Vec<DetectionResult>);

impl From<Vec<DetectionResult>> for DetectionResults {
    fn from(results: Vec<DetectionResult>) -> Self {
        Self(results)
    }
}

impl Deref for DetectionResults {
    type Target = Vec<DetectionResult>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DetectionResults {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl DetectionResults {
    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe {
            slice::from_raw_parts(
                self.0.as_ptr() as *const f32,
                self.0.len() * mem::size_of::<DetectionResult>() / mem::size_of::<f32>(),
            )
        }
    }
}
