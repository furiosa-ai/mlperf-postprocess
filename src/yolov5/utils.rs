use ndarray::Array1;

pub fn centered_box_to_ltrb_bulk(
    pcy: &Array1<f32>,
    pcx: &Array1<f32>,
    pw: &Array1<f32>,
    ph: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
    (pcx - pw * 0.5, pcy - ph * 0.5, pcx + pw * 0.5, pcy + ph * 0.5)
}

/// Detection boxes storing relavant values in place to hack SIMD
pub struct DetectionBoxes {
    pub x1: Array1<f32>,
    pub y1: Array1<f32>,
    pub x2: Array1<f32>,
    pub y2: Array1<f32>,
    pub scores: Array1<f32>,
    pub classes: Array1<usize>,
    pub len: usize,
}

impl DetectionBoxes {
    pub fn new(
        x1: Array1<f32>,
        y1: Array1<f32>,
        x2: Array1<f32>,
        y2: Array1<f32>,
        scores: Array1<f32>,
        classes: Array1<usize>,
    ) -> Self {
        let len = x1.len();
        Self { x1, y1, x2, y2, scores, classes, len }
    }
}
