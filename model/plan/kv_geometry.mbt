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
  layer_count : Int
  kv_head_count : Int
  head_dimension : Int
  dtype : @spec.ModelDType
  layout : KvCacheLayout
} derive(Debug, Eq)

///|
/// Validates persistent KV geometry independently from a graph operation.
pub fn KvCacheGeometry::new(
  layer_count~ : Int,
  kv_head_count~ : Int,
  head_dimension~ : Int,
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
  { layer_count, kv_head_count, head_dimension, dtype, layout }
}

///|
pub fn KvCacheGeometry::layer_count(self : KvCacheGeometry) -> Int {
  self.layer_count
}

///|
pub fn KvCacheGeometry::kv_head_count(self : KvCacheGeometry) -> Int {
  self.kv_head_count
}

///|
pub fn KvCacheGeometry::head_dimension(self : KvCacheGeometry) -> Int {
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
pub fn KvCacheLayout::stable_version(self : KvCacheLayout) -> Int {
  match self {
    LayerMajorSplitKeyValueV1 => 1
  }
}
