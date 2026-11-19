///|
pub(all) enum LayerKind {
  GatedDeltaLinearAttention
  FullAttention
} derive(Eq, Debug)

///|
pub extend LayerKind with Eq::{equal, not_equal}

///|
pub extend LayerKind with @debug.Debug::{to_repr}

///|
pub(all) struct FullAttention {
  query_heads : Int32
  kv_heads : Int32
  head_width : Int32
  output_gate : Bool
} derive(Eq, Debug)

///|
pub extend FullAttention with Eq::{equal, not_equal}

///|
pub extend FullAttention with @debug.Debug::{to_repr}

///|
pub(all) struct RecurrentAttention {
  key_heads : Int32
  key_width : Int32
  value_heads : Int32
  value_width : Int32
  convolution_width : Int32
} derive(Eq, Debug)

///|
pub extend RecurrentAttention with Eq::{equal, not_equal}

///|
pub extend RecurrentAttention with @debug.Debug::{to_repr}

///|
pub(all) struct Experts {
  routed : Int32
  selected : Int32
  intermediate_width : Int32
  shared_width : Int32
} derive(Eq, Debug)

///|
pub extend Experts with Eq::{equal, not_equal}

///|
pub extend Experts with @debug.Debug::{to_repr}

///|
pub suberror GeometryError {
  InvalidGeometry(String)
} derive(Debug)

///|
pub extend GeometryError with @debug.Debug::{to_repr}

///|
/// Shared decoder geometry, independent of vocabulary identities, checkpoint
/// names, storage precision, vision, and kernel admission. This representation
/// describes the repeating linear/full pattern used by the two Qwen3.5-family
/// consumers; it grants no execution or device authority.
pub struct Geometry {
  priv hidden_width : Int32
  priv vocabulary_size : Int32
  priv context_length : Int32
  priv layer_count : Int32
  priv full_interval : Int32
  priv full : FullAttention
  priv recurrent : RecurrentAttention
  priv experts : Experts
}

///|
pub fn Geometry::new(
  hidden_width~ : Int32,
  vocabulary_size~ : Int32,
  context_length~ : Int32,
  layer_count~ : Int32,
  full_interval~ : Int32,
  full~ : FullAttention,
  recurrent~ : RecurrentAttention,
  experts~ : Experts,
) -> Geometry raise GeometryError {
  guard [
      hidden_width,
      vocabulary_size,
      context_length,
      layer_count,
      full_interval,
      full.query_heads,
      full.kv_heads,
      full.head_width,
      recurrent.key_heads,
      recurrent.key_width,
      recurrent.value_heads,
      recurrent.value_width,
      recurrent.convolution_width,
      experts.routed,
      experts.selected,
      experts.intermediate_width,
      experts.shared_width,
    ].all(x => x > 0) &&
    full_interval <= layer_count &&
    full.query_heads % full.kv_heads == 0 &&
    recurrent.value_heads % recurrent.key_heads == 0 &&
    experts.selected <= experts.routed else {
    raise InvalidGeometry("invalid hybrid attention or expert dimensions")
  }
  {
    hidden_width,
    vocabulary_size,
    context_length,
    layer_count,
    full_interval,
    full,
    recurrent,
    experts,
  }
}

///|
pub fn Geometry::hidden_width(self : Geometry) -> Int32 {
  self.hidden_width
}

///|
pub fn Geometry::vocabulary_size(self : Geometry) -> Int32 {
  self.vocabulary_size
}

///|
pub fn Geometry::context_length(self : Geometry) -> Int32 {
  self.context_length
}

///|
pub fn Geometry::layer_count(self : Geometry) -> Int32 {
  self.layer_count
}

///|
pub fn Geometry::full_layer_count(self : Geometry) -> Int32 {
  self.layer_count / self.full_interval
}

///|
pub fn Geometry::recurrent_layer_count(self : Geometry) -> Int32 {
  self.layer_count - self.full_layer_count()
}

///|
pub fn Geometry::layer_kind(self : Geometry, index : Int32) -> LayerKind? {
  if index < 0 || index >= self.layer_count {
    None
  } else if (index + 1) % self.full_interval == 0 {
    Some(FullAttention)
  } else {
    Some(GatedDeltaLinearAttention)
  }
}

///|
pub fn Geometry::full_attention(self : Geometry) -> FullAttention {
  self.full
}

///|
pub fn Geometry::recurrent_attention(self : Geometry) -> RecurrentAttention {
  self.recurrent
}

///|
pub fn Geometry::experts(self : Geometry) -> Experts {
  self.experts
}

///|
fn byte_product(values : Array[Int64]) -> Int64 raise GeometryError {
  for value in values; total = 1L {
    guard value >= 0L && (value == 0L || total <= 9223372036854775807L / value) else {
      raise InvalidGeometry("cache size overflow")
    }
    continue total * value
  } nobreak {
    total
  }
}

///|
/// Logical BF16 K/V payload only: no pages, allocator padding or workspace.
pub fn Geometry::bf16_kv_bytes(
  self : Geometry,
  sequences : Int32,
  tokens : Int32,
) -> Int64 raise GeometryError {
  guard sequences > 0 && tokens >= 0 && tokens <= self.context_length else {
    raise InvalidGeometry("invalid KV capacity")
  }
  byte_product([
    2L,
    2L,
    self.full_layer_count().to_int64(),
    sequences.to_int64(),
    tokens.to_int64(),
    self.full.kv_heads.to_int64(),
    self.full.head_width.to_int64(),
  ])
}

///|
/// Logical FP32 recurrent matrices only; convolution history is separate.
pub fn Geometry::fp32_recurrent_matrix_bytes(
  self : Geometry,
  sequences : Int32,
) -> Int64 raise GeometryError {
  guard sequences > 0 else {
    raise InvalidGeometry("invalid recurrent capacity")
  }
  byte_product([
    4L,
    self.recurrent_layer_count().to_int64(),
    sequences.to_int64(),
    self.recurrent.value_heads.to_int64(),
    self.recurrent.key_width.to_int64(),
    self.recurrent.value_width.to_int64(),
  ])
}
