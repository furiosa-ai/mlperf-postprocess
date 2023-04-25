use indexmap::map::IndexMap;
use itertools::Itertools;
use lazy_static::lazy_static;

use crate::common::proto;

#[derive(Debug, Default, Clone, Copy)]
pub struct TensorIndexer {
    pub ho_stride: usize,
    pub co_stride: usize,
    pub hi_stride: usize,
    pub ci_stride: usize,
    pub w_stride: usize,
    pub hi_limit: usize,
    pub ci_limit: usize,
}

impl<'a> From<&'a proto::shape::TensorShape> for TensorIndexer {
    fn from(shape: &'a proto::shape::TensorShape) -> Self {
        match &shape.inner {
            Some(proto::shape::tensor_shape::Inner::LabeledShape(
                proto::shape::tensor_shape::LabeledShapeInner { inner: Some(labeled_shape) },
            )) => labeled_shape.into(),
            Some(proto::shape::tensor_shape::Inner::LoweredActivationShape(
                proto::shape::tensor_shape::LoweredActivationShapeInner {
                    inner: Some(lowered_activation_shape),
                },
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
        let axis = proto::shape::Axis::from_i32(axis).unwrap();
        match axis {
            proto::shape::Axis::Width => Axis::W,
            proto::shape::Axis::Height => Axis::H,
            proto::shape::Axis::Channel => Axis::C,
            proto::shape::Axis::Batch => Axis::N,
            proto::shape::Axis::Group => Axis::G,
            proto::shape::Axis::WidthOuter => Axis::Wo,
            proto::shape::Axis::HeightOuter => Axis::Ho,
            proto::shape::Axis::ChannelOuter => Axis::Co,
            proto::shape::Axis::BatchOuter => Axis::No,
            proto::shape::Axis::GroupOuter => Axis::Go,
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

impl<'a> From<&'a proto::shape::AxisSizeMap> for AxisSizeMap {
    fn from(axis_size_map: &'a proto::shape::AxisSizeMap) -> Self {
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

impl<'a> From<&'a proto::shape::LabeledShape> for LabeledShape {
    fn from(shape: &'a proto::shape::LabeledShape) -> Self {
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

impl<'a> From<&'a proto::shape::LabeledShape> for TensorIndexer {
    fn from(shape: &'a proto::shape::LabeledShape) -> Self {
        let shape: LabeledShape = shape.into();
        assert!(shape.is_nchw());
        Self {
            ho_stride: 0,
            co_stride: 0,
            hi_stride: shape.width().unwrap(),
            ci_stride: shape.height().unwrap() * shape.width().unwrap(),
            w_stride: 1,
            hi_limit: shape.height().unwrap(),
            ci_limit: shape.channel().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
struct LoweredActivationShape {
    axis_size_map: AxisSizeMap,
    last_dim_size: usize,
    partition_ndim: usize,
}

impl<'a> From<&'a proto::shape::LoweredActivationShape> for LoweredActivationShape {
    fn from(shape: &'a proto::shape::LoweredActivationShape) -> Self {
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

impl<'a> From<&'a proto::shape::LoweredActivationShape> for TensorIndexer {
    fn from(shape: &'a proto::shape::LoweredActivationShape) -> Self {
        let shape: LoweredActivationShape = shape.into();
        if shape.is_nhoco_hcw() {
            Self {
                ho_stride: shape.inner_partitions() * shape.slice_volume(),
                co_stride: shape.slice_volume(),
                hi_stride: shape.slice_channel() * shape.slice_width(),
                ci_stride: shape.slice_width(),
                w_stride: 1,
                hi_limit: shape.slice_height(),
                ci_limit: shape.slice_channel(),
            }
        } else if shape.is_nhoco_hwc() {
            Self {
                ho_stride: shape.inner_partitions() * shape.slice_volume(),
                co_stride: shape.slice_volume(),
                hi_stride: shape.slice_channel() * shape.slice_width(),
                w_stride: shape.slice_channel(),
                ci_stride: 1,
                hi_limit: shape.slice_height(),
                ci_limit: shape.unaligned_slice_channel(),
            }
        } else {
            unimplemented!("Unsupported lowered shape: {:?}", shape);
        }
    }
}

impl TensorIndexer {
    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        let ho = h / self.hi_limit;
        let hi = h % self.hi_limit;
        let co = c / self.ci_limit;
        let ci = c % self.ci_limit;
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
