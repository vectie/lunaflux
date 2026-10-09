///|
fn aligned_device_offset(
  value : Int64,
  alignment : Int64,
  region : Int,
) -> Int64 raise SegmentedDeviceMaterializationError {
  let remainder = value % alignment
  if remainder == 0L {
    return value
  }
  let padding = alignment - remainder
  if value > 9223372036854775807L - padding {
    raise DeviceLayoutOverflow(region)
  }
  value + padding
}

///|
fn validate_source_arenas(
  lengths : ArrayView[Int],
  limits : SegmentedDeviceMaterializationLimits,
) -> ReadOnlyArray[Int] raise SegmentedDeviceMaterializationError {
  if lengths.is_empty() {
    raise EmptySourceArenas
  }
  if lengths.length() > limits.maximum_source_arenas {
    raise TooManySourceArenas
  }
  let owned : Array[Int] = []
  let mut total = 0L
  for index, length in lengths {
    if length <= 0 || length > limits.maximum_host_arena_bytes {
      raise InvalidSourceArena(index)
    }
    let length64 = length.to_int64()
    if length64 > limits.maximum_total_source_bytes - total {
      raise SourceByteBudgetExceeded
    }
    total += length64
    owned.push(length)
  }
  ReadOnlyArray::from_array(owned)
}

///|
fn source_regions_overlap(
  left : SegmentedDeviceTensorRegionInput,
  right : SegmentedDeviceTensorRegionInput,
) -> Bool {
  left.source_arena_ordinal == right.source_arena_ordinal &&
  left.source_offset < right.source_offset + right.byte_count &&
  right.source_offset < left.source_offset + left.byte_count
}

///|
/// Build an immutable one-allocation device layout without touching a device.
pub fn plan_layout(
  source_identity~ : @spec.ModelIdentity,
  source_digest~ : SegmentedHostSourceDigest,
  source_arena_lengths~ : ArrayView[Int],
  regions~ : ArrayView[SegmentedDeviceTensorRegionInput],
  limits~ : SegmentedDeviceMaterializationLimits,
) -> SegmentedDeviceLayout raise SegmentedDeviceMaterializationError {
  let lengths = validate_source_arenas(source_arena_lengths, limits)
  if regions.is_empty() {
    raise EmptyRegions
  }
  if regions.length() > limits.maximum_regions {
    raise TooManyRegions
  }
  let placed : Array[SegmentedDeviceTensorRegion] = []
  let mut device_end = 0L
  let mut materialized = 0L
  let mut previous_tensor_ordinal = -1
  for index, region in regions {
    if (index == 0 && region.tensor_ordinal != 0) ||
      (
        index > 0 &&
        region.tensor_ordinal != previous_tensor_ordinal &&
        region.tensor_ordinal != previous_tensor_ordinal + 1
      ) {
      raise InvalidTensorOrdinal(index)
    }
    guard lengths.get(region.source_arena_ordinal) is Some(arena_length) else {
      raise InvalidSourceRegion(index)
    }
    if region.source_offset > arena_length - region.byte_count {
      raise InvalidSourceRegion(index)
    }
    for earlier in 0..<index {
      if source_regions_overlap(regions[earlier], region) {
        raise OverlappingSourceRegions(earlier, index)
      }
    }
    let offset = aligned_device_offset(
      device_end,
      limits.alignment_bytes,
      index,
    )
    let count = region.byte_count.to_int64()
    if offset > 9223372036854775807L - count ||
      materialized > 9223372036854775807L - count {
      raise DeviceLayoutOverflow(index)
    }
    device_end = offset + count
    materialized += count
    if device_end > limits.maximum_device_arena_bytes {
      raise DeviceArenaTooLarge
    }
    placed.push({
      tensor_ordinal: region.tensor_ordinal,
      source_arena_ordinal: region.source_arena_ordinal,
      source_offset: region.source_offset,
      device_offset: offset,
      byte_count: count,
    })
    previous_tensor_ordinal = region.tensor_ordinal
  }
  let owned_regions = ReadOnlyArray::from_array(placed)
  let digest = canonical_layout_digest(
    source_identity,
    source_digest,
    limits.alignment_bytes,
    device_end,
    materialized,
    lengths,
    owned_regions,
  )
  {
    source_identity,
    source_digest,
    digest,
    alignment_bytes: limits.alignment_bytes,
    device_arena_bytes: device_end,
    materialized_bytes: materialized,
    source_arena_lengths: lengths,
    regions: owned_regions,
  }
}
