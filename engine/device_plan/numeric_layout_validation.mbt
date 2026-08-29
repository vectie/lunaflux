///|
priv struct WeightRegionFacts {
  reference : @plan.TensorRef
  offset_bytes : Int64
  byte_length : Int64
}

///|
fn numeric_tensor_byte_length(
  model_plan : @plan.ModelPlan,
  reference : @plan.TensorRef,
) -> Int64 raise DevicePlanError {
  guard model_plan.tensor(reference) is Some(tensor) else {
    raise MissingLayoutRegion(reference)
  }
  @numeric_contract.storage_byte_length(
    tensor.storage().dtype(),
    tensor.shape(),
  ) catch {
    _ => raise InvalidLayoutRegion(reference, ByteLengthMismatch)
  }
}

///|
fn validate_weight_layout_facts(
  model_plan : @plan.ModelPlan,
  alignment : Int64,
  arena_bytes : Int64,
  materialized_bytes : Int64,
  facts : ArrayView[WeightRegionFacts?],
) -> Unit raise DevicePlanError {
  if arena_bytes <= 0L {
    raise InvalidWeightArena(NonPositiveSize)
  }
  if alignment < 16L || alignment > 4096L || !is_power_of_two(alignment) {
    raise InvalidWeightArena(InvalidAlignment)
  }
  let tensor_count = model_plan.tensor_count()
  if tensor_count <= 0 {
    raise InvalidWeightArena(EmptyRegionSet)
  }
  if facts.length() != tensor_count || facts.length() > MAX_WEIGHT_REGIONS {
    raise InvalidWeightArena(RegionCount)
  }
  let mut previous_end = 0L
  let mut materialized = 0L
  for index, optional in facts {
    let expected_reference = try! @plan.TensorRef::new(index)
    guard optional is Some(region) else {
      raise MissingLayoutRegion(expected_reference)
    }
    if region.reference != expected_reference {
      raise InvalidLayoutRegion(expected_reference, ReferenceMismatch)
    }
    if region.offset_bytes < 0L {
      raise InvalidLayoutRegion(expected_reference, NegativeOffset)
    }
    if region.byte_length <= 0L {
      raise InvalidLayoutRegion(expected_reference, NonPositiveLength)
    }
    if region.offset_bytes % alignment != 0L {
      raise InvalidLayoutRegion(expected_reference, MisalignedOffset)
    }
    if region.offset_bytes < previous_end {
      raise InvalidLayoutRegion(expected_reference, OverlappingPrevious)
    }
    let expected_offset = {
      let remainder = previous_end % alignment
      if remainder == 0L {
        previous_end
      } else {
        let padding = alignment - remainder
        if previous_end > MAX_INT64 - padding {
          raise InvalidLayoutRegion(expected_reference, EndOverflow)
        }
        previous_end + padding
      }
    }
    if region.offset_bytes != expected_offset {
      raise InvalidLayoutRegion(expected_reference, NonCanonicalOffset)
    }
    if region.offset_bytes > MAX_INT64 - region.byte_length {
      raise InvalidLayoutRegion(expected_reference, EndOverflow)
    }
    let end = region.offset_bytes + region.byte_length
    if end > arena_bytes {
      raise InvalidLayoutRegion(expected_reference, OutsideArena)
    }
    if region.byte_length !=
      numeric_tensor_byte_length(model_plan, expected_reference) {
      raise InvalidLayoutRegion(expected_reference, ByteLengthMismatch)
    }
    if materialized > MAX_INT64 - region.byte_length {
      raise InvalidLayoutRegion(expected_reference, EndOverflow)
    }
    materialized += region.byte_length
    previous_end = end
  }
  if materialized != materialized_bytes {
    raise InvalidWeightArena(MaterializedSizeMismatch)
  }
  if previous_end != arena_bytes {
    raise InvalidWeightArena(TerminalExtentMismatch)
  }
}

///|
fn validate_weight_layout(
  model_plan : @plan.ModelPlan,
  layout : @device_materialize.DeviceWeightLayout,
) -> Unit raise DevicePlanError {
  if layout.length() <= 0 {
    raise InvalidWeightArena(EmptyRegionSet)
  }
  if layout.length() != model_plan.tensor_count() ||
    layout.length() > MAX_WEIGHT_REGIONS {
    raise InvalidWeightArena(RegionCount)
  }
  let facts : Array[WeightRegionFacts?] = []
  for index in 0..<layout.length() {
    let reference = try! @plan.TensorRef::new(index)
    match layout.region(reference) {
      Some(region) =>
        facts.push(
          Some({
            reference: region.reference(),
            offset_bytes: region.offset_bytes(),
            byte_length: region.byte_length(),
          }),
        )
      None => facts.push(None)
    }
  }
  validate_weight_layout_facts(
    model_plan,
    layout.alignment_bytes(),
    layout.arena_bytes(),
    layout.materialized_bytes(),
    facts,
  )
}
