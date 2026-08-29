///|
const MAX_INT64 : Int64 = 9223372036854775807L

///|
const MIN_ALIGNMENT : Int64 = 16L

///|
const MAX_ALIGNMENT : Int64 = 4096L

///|
fn is_power_of_two(value : Int64) -> Bool {
  value > 0L && (value & (value - 1L)) == 0L
}

///|
fn checked_align(
  value : Int64,
  alignment : Int64,
  reference : Int,
) -> Int64 raise DeviceWeightError {
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    let padding = alignment - remainder
    if value > MAX_INT64 - padding {
      raise LayoutSizeOverflow(reference~)
    }
    value + padding
  }
}

///|
/// Plans one contiguous arena in semantic `TensorRef` order. Every tensor
/// begins at an exact multiple of `alignment_bytes`; only inter-tensor padding
/// contributes to `arena_bytes`.
pub fn plan_layout(
  bindings : @llama_weights.LlamaWeightBindings,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeightLayout raise DeviceWeightError {
  if alignment_bytes < MIN_ALIGNMENT ||
    alignment_bytes > MAX_ALIGNMENT ||
    !is_power_of_two(alignment_bytes) {
    raise InvalidAlignment
  }
  if max_arena_bytes <= 0L {
    raise InvalidArenaLimit
  }
  if bindings.length() == 0 {
    raise EmptyBindings
  }
  let regions : Array[DeviceWeightRegion] = []
  let mut end = 0L
  let mut materialized = 0L
  for reference in 0..<bindings.length() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    guard bindings.binding(tensor_reference) is Some(binding) else {
      raise MissingBinding(reference~)
    }
    if binding.reference() != tensor_reference {
      raise BindingOrderMismatch(reference~)
    }
    let offset = checked_align(end, alignment_bytes, reference)
    let byte_length = binding.tensor().byte_length()
    if byte_length <= 0L || offset > MAX_INT64 - byte_length {
      raise LayoutSizeOverflow(reference~)
    }
    if materialized > MAX_INT64 - byte_length {
      raise LayoutSizeOverflow(reference~)
    }
    end = offset + byte_length
    materialized = materialized + byte_length
    if end > max_arena_bytes {
      raise ArenaTooLarge
    }
    regions.push({
      reference: tensor_reference,
      offset_bytes: offset,
      byte_length,
    })
  }
  if materialized != bindings.materialized_bytes() {
    raise LayoutSizeOverflow(reference=bindings.length() - 1)
  }
  {
    identity: bindings.identity(),
    numeric_schema_digest: None,
    alignment_bytes,
    arena_bytes: end,
    materialized_bytes: materialized,
    regions: ReadOnlyArray::from_array(regions),
  }
}
