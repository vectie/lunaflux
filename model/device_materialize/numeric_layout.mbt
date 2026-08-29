///|
const MAX_NUMERIC_LAYOUT_TENSORS : Int = 65536

///|
/// Plans one inert contiguous arena from the complete canonical numeric tensor
/// table embedded in `model_plan`.
///
/// Every parameter and metadata tensor is placed in exact `TensorRef` order.
/// Byte size comes only from the backend-free numeric storage contract. This
/// function opens no file or device resource and makes no readiness claim.
pub fn plan_numeric_layout(
  model_plan : @plan.ModelPlan,
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
  let tensors = model_plan.numeric_binding().tensors()
  if tensors.is_empty() {
    raise EmptyBindings
  }
  if tensors.length() > MAX_NUMERIC_LAYOUT_TENSORS {
    raise LayoutSizeOverflow(reference=tensors.length())
  }
  if tensors.length() != model_plan.tensor_count() {
    raise LayoutSizeOverflow(reference=tensors.length())
  }
  let regions : Array[DeviceWeightRegion] = []
  let mut end = 0L
  let mut materialized = 0L
  for index, tensor in tensors {
    let reference = try! @plan.TensorRef::new(index)
    if tensor.reference() != reference {
      raise BindingOrderMismatch(reference=index)
    }
    let offset = checked_align(end, alignment_bytes, index)
    let byte_length = @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    ) catch {
      _ => raise LayoutSizeOverflow(reference=index)
    }
    if byte_length <= 0L || offset > MAX_INT64 - byte_length {
      raise LayoutSizeOverflow(reference=index)
    }
    if materialized > MAX_INT64 - byte_length {
      raise LayoutSizeOverflow(reference=index)
    }
    let next_end = offset + byte_length
    let next_materialized = materialized + byte_length
    if next_end > max_arena_bytes {
      raise ArenaTooLarge
    }
    regions.push({ reference, offset_bytes: offset, byte_length })
    end = next_end
    materialized = next_materialized
  }
  {
    identity: model_plan.identity(),
    numeric_schema_digest: Some(model_plan.numeric_binding().digest()),
    alignment_bytes,
    arena_bytes: end,
    materialized_bytes: materialized,
    regions: ReadOnlyArray::from_array(regions),
  }
}
