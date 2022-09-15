use indexmap::map::IndexMap;
use itertools::Itertools;
use lazy_static::lazy_static;

use crate::common::proto;
use crate::common::proto::common::{tensor_shape, TensorShape};

#[derive(Debug, Default, Clone, Copy)]
pub struct LoweredShape {
    ho_stride: usize,
    co_stride: usize,
    hi_stride: usize,
    ci_stride: usize,
    w_stride: usize,
    slice_height: usize,
    slice_channel: usize,
}

impl<'a> From<&'a TensorShape> for LoweredShape {
    fn from(shape: &'a TensorShape) -> Self {
        match &shape.inner {
            Some(tensor_shape::Inner::LabeledShape(tensor_shape::LabeledShapeInner {
                inner: Some(labeled_shape),
            })) => labeled_shape.into(),
            Some(tensor_shape::Inner::LoweredActivationShape(
                tensor_shape::LoweredActivationShapeInner { inner: Some(lowered_activation_shape) },
            )) => lowered_activation_shape.into(),
            _ => unimplemented!("Unsupported lowered shape: {:?}", shape),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Axis {
    W,
    H,
    C,
    N,
    G,
    Wo,
    Ho,
    Co,
    No,
    Go,
}

impl From<i32> for Axis {
    fn from(axis: i32) -> Self {
        let axis = proto::common::Axis::from_i32(axis).unwrap();
        match axis {
            proto::common::Axis::Width => Axis::W,
            proto::common::Axis::Height => Axis::H,
            proto::common::Axis::Channel => Axis::C,
            proto::common::Axis::Batch => Axis::N,
            proto::common::Axis::Group => Axis::G,
            proto::common::Axis::WidthOuter => Axis::Wo,
            proto::common::Axis::HeightOuter => Axis::Ho,
            proto::common::Axis::ChannelOuter => Axis::Co,
            proto::common::Axis::BatchOuter => Axis::No,
            proto::common::Axis::GroupOuter => Axis::Go,
        }
    }
}

lazy_static! {
    static ref NCHW: Vec<Axis> = vec![Axis::N, Axis::C, Axis::H, Axis::W];
    static ref LOWERED_HCW: Vec<Axis> =
        vec![Axis::N, Axis::Ho, Axis::Co, Axis::H, Axis::C, Axis::W];
    static ref LOWERED_HWC: Vec<Axis> =
        vec![Axis::N, Axis::Ho, Axis::Co, Axis::H, Axis::W, Axis::C];
}

#[derive(Debug, Default, Clone)]
struct AxisSizeMap(pub IndexMap<Axis, usize>);

impl<'a> From<&'a proto::common::AxisSizeMap> for AxisSizeMap {
    fn from(axis_size_map: &'a proto::common::AxisSizeMap) -> Self {
        let axis_size_map = axis_size_map
            .inner
            .iter()
            .map(|axis_size| ((axis_size.k).into(), axis_size.v.try_into().unwrap()))
            .collect::<IndexMap<Axis, usize>>();
        Self(axis_size_map)
    }
}

#[derive(Debug, Default, Clone)]
struct LabeledShape {
    axis_size_map: AxisSizeMap,
}

impl<'a> From<&'a proto::common::LabeledShape> for LabeledShape {
    fn from(shape: &'a proto::common::LabeledShape) -> Self {
        Self { axis_size_map: shape.axis_size_map.as_ref().unwrap().into() }
    }
}

impl LabeledShape {
    fn is_nchw(&self) -> bool {
        &self.axis_size_map.0.keys().cloned().collect::<Vec<Axis>>() == &NCHW as &Vec<Axis>
    }

    fn width(&self) -> Option<usize> {
        self.axis_size_map.0.get(&Axis::W).cloned()
    }

    fn height(&self) -> Option<usize> {
        self.axis_size_map.0.get(&Axis::H).cloned()
    }

    fn channel(&self) -> Option<usize> {
        self.axis_size_map.0.get(&Axis::C).cloned()
    }
}

impl<'a> From<&'a proto::common::LabeledShape> for LoweredShape {
    fn from(shape: &'a proto::common::LabeledShape) -> Self {
        let shape: LabeledShape = shape.into();
        assert!(shape.is_nchw());
        Self {
            ho_stride: 0,
            co_stride: 0,
            hi_stride: shape.width().unwrap(),
            ci_stride: shape.height().unwrap() * shape.width().unwrap(),
            w_stride: 1,
            slice_height: shape.height().unwrap(),
            slice_channel: shape.channel().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
struct LoweredActivationShape {
    axis_size_map: AxisSizeMap,
    last_dim_size: usize,
    partition_ndim: usize,
}

impl<'a> From<&'a proto::common::LoweredActivationShape> for LoweredActivationShape {
    fn from(shape: &'a proto::common::LoweredActivationShape) -> Self {
        Self {
            axis_size_map: shape.axis_size_map.as_ref().unwrap().into(),
            #[cfg(feature = "legacy-npu-tools")]
            last_dim_size: shape.last_dim_size as usize,
            #[cfg(not(feature = "legacy-npu-tools"))]
            last_dim_size: shape.last_dim_alignment as usize,
            partition_ndim: shape.partition_ndim as usize,
        }
    }
}

impl LoweredActivationShape {
    fn is_nhoco_hwc(&self) -> bool {
        &self.axis_size_map.0.keys().cloned().collect::<Vec<Axis>>() == &LOWERED_HWC as &Vec<Axis>
    }

    fn is_nhoco_hcw(&self) -> bool {
        &self.axis_size_map.0.keys().cloned().collect::<Vec<Axis>>() == &LOWERED_HCW as &Vec<Axis>
    }

    fn inner_partitions(&self) -> usize {
        *self.axis_size_map.0.get_index(self.partition_ndim - 1).unwrap().1
    }

    fn iter_slice_axis_size(&self) -> impl Iterator<Item = (&Axis, &usize)> {
        self.axis_size_map.0.iter().dropping(self.partition_ndim)
    }

    fn slice_volume(&self) -> usize {
        self.iter_slice_axis_size().fold(1, |acc, (_, &size)| size * acc)
    }

    fn slice_width(&self) -> usize {
        self.axis_size_map.0.get(&Axis::W).cloned().unwrap()
    }

    fn slice_channel(&self) -> usize {
        self.axis_size_map.0.get(&Axis::C).cloned().unwrap()
    }

    fn slice_height(&self) -> usize {
        self.axis_size_map.0.get(&Axis::H).cloned().unwrap()
    }

    fn unaligned_slice_channel(&self) -> usize {
        if self.is_last(Axis::C) {
            self.last_dim_size
        } else {
            self.slice_channel()
        }
    }

    fn last_axis(&self) -> Axis {
        let index = self.axis_size_map.0.len() - 1;
        *self.axis_size_map.0.get_index(index).unwrap().0
    }

    fn is_last(&self, axis: Axis) -> bool {
        self.last_axis() == axis
    }
}

impl<'a> From<&'a proto::common::LoweredActivationShape> for LoweredShape {
    fn from(shape: &'a proto::common::LoweredActivationShape) -> Self {
        let shape: LoweredActivationShape = shape.into();
        if shape.is_nhoco_hcw() {
            Self {
                ho_stride: shape.inner_partitions() * shape.slice_volume(),
                co_stride: shape.slice_volume(),
                hi_stride: shape.slice_channel() * shape.slice_width(),
                ci_stride: shape.slice_width(),
                w_stride: 1,
                slice_height: shape.slice_height(),
                slice_channel: shape.slice_channel(),
            }
        } else if shape.is_nhoco_hwc() {
            Self {
                ho_stride: shape.inner_partitions() * shape.slice_volume(),
                co_stride: shape.slice_volume(),
                hi_stride: shape.slice_channel() * shape.slice_width(),
                w_stride: shape.slice_channel(),
                ci_stride: 1,
                slice_height: shape.slice_height(),
                slice_channel: shape.unaligned_slice_channel(),
            }
        } else {
            unimplemented!("Unsupported lowered shape: {:?}", shape);
        }
    }
}

impl LoweredShape {
    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        let ho = h / self.slice_height;
        let hi = h % self.slice_height;
        let co = c / self.slice_channel;
        let ci = c % self.slice_channel;
        ho * self.ho_stride
            + co * self.co_stride
            + hi * self.hi_stride
            + ci * self.ci_stride
            + w * self.w_stride
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    height: usize,
    width: usize,
}

impl Shape {
    pub fn new(h: usize, w: usize) -> Self {
        Self { height: h, width: w }
    }

    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        c * self.height * self.width + h * self.width + w
    }
}
