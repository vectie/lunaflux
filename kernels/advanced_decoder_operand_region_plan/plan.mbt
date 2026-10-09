///|
fn regions_overlap(
  left : AdvancedDecoderOperandRegion,
  right : AdvancedDecoderOperandRegion,
) -> Bool {
  left.allocation_identity_sha256 == right.allocation_identity_sha256 &&
  left.device_offset < right.end_offset &&
  right.device_offset < left.end_offset
}

///|
/// Binds exact admitted operand ABIs to inert allocation-relative regions.
/// Allocation identities, offsets, and capacities are metadata only; no live
/// device pointer, allocation ownership, loader, or launch authority enters
/// this value.
pub fn plan_advanced_decoder_operand_regions(
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  inputs : ArrayView[AdvancedDecoderOperandRegionInput],
  limits : AdvancedDecoderOperandRegionPlanLimits,
) -> AdvancedDecoderOperandRegionPlan raise AdvancedDecoderOperandRegionPlanError {
  if operands.is_empty() ||
    operands.length() != inputs.length() ||
    operands.length() > limits.maximum_operands {
    raise OperandCountMismatch
  }
  let regions : Array[AdvancedDecoderOperandRegion] = []
  for ordinal, operand in operands {
    let input = inputs[ordinal]
    if input.operand_ordinal != ordinal {
      raise OperandOrdinalMismatch(ordinal)
    }
    if input.role != operand.role() {
      raise OperandRoleMismatch(ordinal)
    }
    let byte_count = operand.byte_count()
    let alignment = operand.alignment()
    if input.device_offset % alignment != 0L {
      raise OperandAlignmentMismatch(ordinal)
    }
    if input.allocation_capacity_bytes > limits.maximum_allocation_bytes {
      raise AllocationCapacityMismatch(ordinal)
    }
    if input.device_offset > input.allocation_capacity_bytes - byte_count {
      raise RegionUndersized(ordinal)
    }
    let current : AdvancedDecoderOperandRegion = {
      operand_ordinal: ordinal,
      role: input.role,
      byte_count,
      alignment,
      allocation_identity_sha256: input.allocation_identity_sha256,
      device_offset: input.device_offset,
      end_offset: input.device_offset + byte_count,
      allocation_capacity_bytes: input.allocation_capacity_bytes,
    }
    for earlier, region in regions {
      if region.allocation_identity_sha256 == input.allocation_identity_sha256 &&
        region.allocation_capacity_bytes != input.allocation_capacity_bytes {
        raise AllocationCapacityMismatch(ordinal)
      }
      if regions_overlap(region, current) {
        raise RegionOverlap(earlier, ordinal)
      }
    }
    regions.push(current)
  }
  let region_view = ReadOnlyArray::from_array(regions)
  {
    regions: region_view,
    digest: canonical_operand_region_plan_digest(
      region_view,
      limits.maximum_canonical_bytes,
    ),
  }
}
