///|
fn luna_specialization_static_operation(
  plan : @device_plan.StaticDevicePlan,
  id : @plan.OperationId,
) -> @device_plan.StaticOperation? {
  for operation in plan.operations() {
    if operation.id() == id {
      return Some(operation)
    }
  }
  None
}

///|
fn luna_specialization_plan_operation(
  plan : @plan.ModelPlan,
  id : @plan.OperationId,
) -> @plan.PlanOperation? {
  for operation in plan.operations() {
    if operation.id() == id {
      return Some(operation)
    }
  }
  None
}

///|
fn luna_specialization_binding(
  catalog : @catalog.ResolvedKernelCatalog,
  id : @plan.OperationId,
) -> @catalog.KernelBinding? {
  for binding in catalog.bindings() {
    if binding.operation_id() == id {
      return Some(binding)
    }
  }
  None
}

///|
fn luna_specialization_region(
  space : LunaSpecializedRegionSpace,
  semantic_id : Int,
  offset_bytes : Int64,
  byte_length : Int64,
  alignment_bytes : Int64,
) -> LunaSpecializedRegion raise LunaSpecializationError {
  if offset_bytes < 0L ||
    byte_length <= 0L ||
    alignment_bytes <= 0L ||
    (alignment_bytes & (alignment_bytes - 1L)) != 0L ||
    offset_bytes % alignment_bytes != 0L ||
    offset_bytes > 9223372036854775807L - byte_length {
    raise Invalid(TensorOffset)
  }
  { space, semantic_id, offset_bytes, byte_length, alignment_bytes }
}

///|
fn collect_luna_specialization_profile_region(
  regions : Array[LunaSpecializedRegion],
  semantic_id : Int,
  region : @device_profile.ExecutionRegionView,
) -> Unit raise LunaSpecializationError {
  if region.logical_bytes() == 0L {
    return
  }
  let space = match region.space() {
    TokenInput => LunaTokenInput
    WeightArena => LunaWeightArena
    ActivationWorkspaceArena => LunaActivationWorkspaceArena
  }
  regions.push(
    luna_specialization_region(
      space,
      semantic_id,
      region.offset_bytes(),
      region.logical_bytes(),
      region.alignment_bytes(),
    ),
  )
}

///|
fn collect_luna_specialization_regions(
  profile : @device_profile.OperationExecutionProfile,
  maximum : Int,
) -> ReadOnlyArray[LunaSpecializedRegion] raise LunaSpecializationError {
  let regions : Array[LunaSpecializedRegion] = []
  for index, input in profile.inputs() {
    match input {
      TokenIds(region) =>
        collect_luna_specialization_profile_region(
          regions,
          100000 + index,
          region,
        )
      Weight(reference, region) =>
        collect_luna_specialization_profile_region(
          regions,
          reference.as_int(),
          region,
        )
      Activation(value) =>
        collect_luna_specialization_profile_region(
          regions,
          value.reference().as_int(),
          value.region(),
        )
    }
  }
  for value in profile.outputs() {
    collect_luna_specialization_profile_region(
      regions,
      value.reference().as_int(),
      value.region(),
    )
  }
  collect_luna_specialization_profile_region(
    regions,
    200000,
    profile.workspace(),
  )
  if regions.length() > maximum {
    raise Invalid(TensorLayout)
  }
  ReadOnlyArray::from_array(regions)
}

///|
fn collect_luna_semantic_bits(
  model_plan : @plan.ModelPlan,
) -> (UInt64, UInt64) raise LunaSpecializationError {
  let mut rope_bits = 0UL
  let mut rms_bits = 0UL
  for operation in model_plan.operations() {
    match operation.shape() {
      RotaryShape(rope_theta~, ..) => {
        let bits = rope_theta.reinterpret_as_uint64()
        if rope_bits != 0UL && rope_bits != bits {
          raise Mismatch(Rope)
        }
        rope_bits = bits
      }
      NormalizationShape(epsilon~, ..) => {
        let bits = epsilon.reinterpret_as_uint64()
        if rms_bits != 0UL && rms_bits != bits {
          raise Mismatch(OperationShape)
        }
        rms_bits = bits
      }
      _ => ()
    }
  }
  if rope_bits == 0UL || rms_bits == 0UL {
    raise Invalid(OperationShape)
  }
  (rope_bits, rms_bits)
}
