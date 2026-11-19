///|
/// Canonical v1 physical ordering of persistent key/value cache elements.
///
/// This is semantic model-plan data, not a device allocation or page-size
/// choice. A later storage plan combines it with configured page capacity.
pub(all) enum KvCacheLayout {
  LayerMajorSplitKeyValueV1
} derive(Debug, Eq)

///|
/// Immutable architecture-neutral geometry required by cached attention.
///
/// Page size, page count, byte strides, device placement, and resource
/// ownership belong to later KV storage planning. This value records only the
/// model facts needed to derive those decisions without model-family checks.
pub struct KvCacheGeometry {
  priv layer_count : Int32
  priv kv_head_count : Int32
  priv head_dimension : Int32
  priv dtype : @spec.ModelDType
  priv layout : KvCacheLayout
} derive(Debug, Eq)

///|
/// Validates persistent KV geometry independently from a graph operation.
pub fn KvCacheGeometry::new(
  layer_count~ : Int32,
  kv_head_count~ : Int32,
  head_dimension~ : Int32,
  dtype~ : @spec.ModelDType,
  layout~ : KvCacheLayout,
) -> KvCacheGeometry raise PlanValidationError {
  if layer_count <= 0 || kv_head_count <= 0 || head_dimension <= 0 {
    raise InvalidShape(KvCachePositiveDimension)
  }
  if kv_head_count > 2147483647 / head_dimension {
    raise InvalidShape(KvCacheDimensionOverflow)
  }
  if dtype != BFloat16 {
    raise InvalidShape(UnsupportedKvCacheDType)
  }
  match layout {
    LayerMajorSplitKeyValueV1 => ()
  }
  { layer_count, kv_head_count, head_dimension, dtype, layout, }
}

///|
pub fn KvCacheGeometry::layer_count(self : KvCacheGeometry) -> Int32 {
  self.layer_count
}

///|
pub fn KvCacheGeometry::kv_head_count(self : KvCacheGeometry) -> Int32 {
  self.kv_head_count
}

///|
pub fn KvCacheGeometry::head_dimension(self : KvCacheGeometry) -> Int32 {
  self.head_dimension
}

///|
pub fn KvCacheGeometry::dtype(self : KvCacheGeometry) -> @spec.ModelDType {
  self.dtype
}

///|
pub fn KvCacheGeometry::layout(self : KvCacheGeometry) -> KvCacheLayout {
  self.layout
}

///|
pub fn KvCacheLayout::stable_version(self : KvCacheLayout) -> Int32 {
  match self {
    LayerMajorSplitKeyValueV1 => 1
  }
}

///|
pub extend KvCacheLayout with @debug.Debug::{to_repr}

///|
pub extend KvCacheLayout with Eq::{not_equal, equal}

///|
pub extend KvCacheGeometry with @debug.Debug::{to_repr}

///|
pub extend KvCacheGeometry with Eq::{not_equal, equal}
