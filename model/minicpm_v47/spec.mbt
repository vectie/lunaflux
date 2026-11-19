///|
pub suberror ConfigError {
  InvalidConfig(String)
} derive(Debug)

///|
pub extend ConfigError with @debug.Debug::{to_repr}

///|
pub(all) enum Downsample {
  FourX
  SixteenX
} derive(Debug, Eq)

///|
pub extend Downsample with @debug.Debug::{to_repr}

///|
pub extend Downsample with Eq::{equal, not_equal}

///|
pub(all) struct VisionGeometry {
  hidden_width : Int32
  intermediate_width : Int32
  layers : Int32
  heads : Int32
  channels : Int32
  position_grid_side : Int32
  patch_size : Int32
  compression_after_layer : Int32
  window_side : Int32
  merge_rounds : Int32
}

///|
pub(all) struct Markers {
  image : Int32
  video : Int32
  image_start : Int32
  image_end : Int32
  slice_start : Int32
  slice_end : Int32
  newline : Int32
}

///|
pub(all) struct Rotary {
  theta : Double
  partial_factor : Double
  temporal_sections : Int32
  height_sections : Int32
  width_sections : Int32
}

///|
/// Validated semantics for the selected BF16 MiniCPM-V 4.7 35B-A3B checkpoint.
/// Tensor inspection, numerical plans and CUDA admission are subsequent steps.
/// The outer model retains vision; no conversion to a text-only Qwen artifact.
pub struct Spec {
  priv decoder : @hybrid.Geometry
  priv vision : VisionGeometry
  priv markers : Markers
  priv rotary : Rotary
  priv downsample : Downsample
}

///|
pub fn Spec::decoder(self : Spec) -> @hybrid.Geometry {
  self.decoder
}

///|
pub fn Spec::vision(self : Spec) -> VisionGeometry {
  self.vision
}

///|
pub fn Spec::markers(self : Spec) -> Markers {
  self.markers
}

///|
pub fn Spec::rotary(self : Spec) -> Rotary {
  self.rotary
}

///|
pub fn Spec::downsample(self : Spec) -> Downsample {
  self.downsample
}
