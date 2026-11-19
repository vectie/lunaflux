///|
pub(all) suberror LayoutError {
  UnsupportedModel
  UnsupportedContext
  InsufficientMemory
  InvalidIndex
}

///|
pub(all) struct Region {
  address : UInt64
  bytes : UInt64
}

///|
pub struct Layout {
  priv content_label : @spec.ContentDigest
  priv context : Int32
  priv weights : ReadOnlyArray[Region]
  priv dimensions : ReadOnlyArray[ReadOnlyArray[Int64]]
  priv norms : ReadOnlyArray[Region?]
  priv kv_base : UInt64
  priv kv_bytes : UInt64
  priv workspace : Region
  priv reserve : Region
  priv free_bytes : UInt64
  priv weight_bytes : UInt64
}

///|
fn aligned(value : UInt64) -> UInt64 {
  (value + 4095UL) / 4096UL * 4096UL
}

///|
/// Startup-only full-model placement. The source Qwen spec's BF16 numerical
/// operation schema is not reused: this target explicitly has FP32 KV and
/// activations. Only its validated geometry and source tensor inventory apply.
pub fn prepare(
  device : @lunachip.DeviceContract,
  metadata : @qwen.Qwen3ModelMetadata,
  context~ : Int32 = 2048,
  align_matrices_1k~ : Bool = false,
  workspace_bytes~ : UInt64 = 67108864UL,
  reserve_bytes~ : UInt64 = 67108864UL,
) -> Layout raise LayoutError {
  let spec = metadata.spec()
  if spec.vocabulary_size() != 151936 ||
    spec.hidden_size() != 1024 ||
    spec.intermediate_size() != 3072 ||
    spec.layer_count() != 28 ||
    spec.query_head_count() != 16 ||
    spec.kv_head_count() != 8 ||
    spec.head_dimension() != 128 ||
    spec.rope_theta() != 1000000.0 ||
    spec.rms_norm_epsilon() != 0.000001 {
    raise UnsupportedModel
  }
  if context < 1 || context > 2048 || context > spec.context_length() {
    raise UnsupportedContext
  }
  let capacity = device.capacity_bytes()
  // Reservation is a preparation budget, not a proof of the future complete
  // lowering's workspace requirement. Retain a conservative minimum.
  if workspace_bytes < 1048576UL ||
    reserve_bytes < 1048576UL ||
    workspace_bytes > capacity ||
    reserve_bytes > capacity {
    raise InsufficientMemory
  }
  let weights : Array[Region] = []
  let dimensions : Array[ReadOnlyArray[Int64]] = []
  let norms : Array[Region?] = []
  let schema = spec.numeric_schema()
  let mut cursor = 0UL
  let mut weight_bytes = 0UL
  for tensor in schema.tensors() {
    let bytes = tensor.shape().element_count().reinterpret_as_uint64() * 2UL
    if align_matrices_1k && tensor.shape().rank() == 2 {
      cursor = (cursor + 1023UL) / 1024UL * 1024UL
    }
    weights.push({ address: cursor, bytes, })
    dimensions.push(tensor.shape().dimensions())
    cursor += bytes
    weight_bytes += bytes
  }
  for tensor in schema.tensors() {
    if tensor.shape().rank() == 1 {
      let bytes = tensor.shape().element_count().reinterpret_as_uint64() * 4UL
      norms.push(Some({ address: cursor, bytes, }))
      cursor += bytes
    } else {
      norms.push(None)
    }
  }
  let kv_base = aligned(cursor)
  let kv_bytes = 28UL * 2UL * context.to_uint64() * 1024UL * 4UL
  cursor = aligned(kv_base + kv_bytes)
  let workspace : Region = { address: cursor, bytes: workspace_bytes, }
  cursor = aligned(cursor + workspace_bytes)
  let reserve : Region = { address: cursor, bytes: reserve_bytes, }
  cursor = aligned(cursor + reserve_bytes)
  if cursor > capacity {
    raise InsufficientMemory
  }
  {
    content_label: metadata.content_digest(),
    context,
    weights: ReadOnlyArray::from_array(weights),
    dimensions: ReadOnlyArray::from_array(dimensions),
    norms: ReadOnlyArray::from_array(norms),
    kv_base,
    kv_bytes,
    workspace,
    reserve,
    free_bytes: capacity - cursor,
    weight_bytes,
  }
}

///|
pub fn Layout::content_label(self : Self) -> @spec.ContentDigest {
  self.content_label
}

///|
pub fn Layout::context(self : Self) -> Int32 {
  self.context
}

///|
pub fn Layout::weight_count(self : Self) -> Int32 {
  self.weights.length()
}

///|
pub fn Layout::weight_bytes(self : Self) -> UInt64 {
  self.weight_bytes
}

///|
pub fn Layout::weight(self : Self, ordinal : Int32) -> Region? {
  self.weights.get(ordinal)
}

///|
pub fn Layout::dimensions(
  self : Self,
  ordinal : Int32,
) -> ReadOnlyArray[Int64]? {
  self.dimensions.get(ordinal)
}

///|
pub fn Layout::expanded_norm(self : Self, ordinal : Int32) -> Region? {
  self.norms.get(ordinal).bind(value => value)
}

///|
pub fn Layout::lm_head(self : Self) -> Region {
  self.weights[0]
}

///|
pub fn Layout::workspace(self : Self) -> Region {
  self.workspace
}

///|
pub fn Layout::reserve(self : Self) -> Region {
  self.reserve
}

///|
pub fn Layout::free_bytes(self : Self) -> UInt64 {
  self.free_bytes
}

///|
pub fn Layout::kv_bytes(self : Self) -> UInt64 {
  self.kv_bytes
}

///|
/// Layer-major, then K/V, then token, head and FP32 head element.
pub fn Layout::kv_head(
  self : Self,
  layer : Int32,
  value : Bool,
  position : Int32,
  head : Int32,
) -> Region raise LayoutError {
  if layer < 0 ||
    layer >= 28 ||
    position < 0 ||
    position >= self.context ||
    head < 0 ||
    head >= 8 {
    raise InvalidIndex
  }
  let bank = layer * 2 + (if value { 1 } else { 0 })
  let offset = (
      (bank.to_uint64() * self.context.to_uint64() + position.to_uint64()) * 8UL +
      head.to_uint64()
    ) *
    512UL
  { address: self.kv_base + offset, bytes: 512UL, }
}
