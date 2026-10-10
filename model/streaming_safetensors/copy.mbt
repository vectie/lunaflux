///|
fn validate_copy_requests(
  inspection : StreamingSafetensorsInspection,
  destination : FixedArray[Byte],
  requests : ArrayView[StreamingSafetensorsCopyRequest],
  limits : StreamingSafetensorsLimits,
) -> Unit raise StreamingSafetensorsError {
  if requests.length() > limits.maximum_copy_requests {
    raise CopyBudgetExceeded
  }
  let mut total = 0L
  for index, request in requests {
    guard inspection.shards.get(request.shard_ordinal) is Some(shard) else {
      raise InvalidCopyRequest(index)
    }
    guard shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) else {
      raise InvalidCopyRequest(index)
    }
    let count = request.byte_count.to_int64()
    if request.tensor_byte_offset > tensor.byte_count() - count {
      raise CopyRangeOutsideTensor(index)
    }
    if request.destination_offset > destination.length() - request.byte_count {
      raise CopyRangeOutsideDestination(index)
    }
    if total > limits.maximum_copy_bytes - count {
      raise CopyBudgetExceeded
    }
    total += count
    for earlier in 0..<index {
      let previous = requests[earlier]
      if previous.destination_offset <
        request.destination_offset + request.byte_count &&
        request.destination_offset <
        previous.destination_offset + previous.byte_count {
        raise OverlappingDestinationRanges(earlier, index)
      }
    }
  }
}

///|
fn copy_open_shard(
  shard_ordinal : Int,
  file : @approved_fs.ApprovedFile,
  shard : InspectedShard,
  destination : FixedArray[Byte],
  requests : ArrayView[StreamingSafetensorsCopyRequest],
  visit : (StreamingSafetensorsTensorMetadata, StreamingSafetensorsVisitedRange) -> Unit raise StreamingSafetensorsError,
) -> Unit raise StreamingSafetensorsError {
  let initial_stamp = stamp_file(shard_ordinal, file)
  if initial_stamp != shard.stamp ||
    initial_stamp.byte_count() != shard.metadata.file_byte_count {
    raise InspectionReplayMismatch(shard_ordinal)
  }
  for request in requests {
    if request.shard_ordinal != shard_ordinal {
      continue
    }
    guard shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) else {
      raise InvalidCopyRequest(-1)
    }
    let source_position = shard.metadata.payload_offset +
      tensor.data_start +
      request.tensor_byte_offset
    read_exactly(
      shard_ordinal,
      file,
      destination,
      request.destination_offset,
      request.byte_count,
      source_position,
    )
  }
  if stamp_file(shard_ordinal, file) != initial_stamp {
    raise FileChanged(shard_ordinal)
  }
  for request in requests {
    if request.shard_ordinal != shard_ordinal {
      continue
    }
    guard shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) else {
      raise InvalidCopyRequest(-1)
    }
    visit(tensor, {
      shard_ordinal,
      tensor_ordinal: request.tensor_ordinal,
      tensor_byte_offset: request.tensor_byte_offset,
      destination_offset: request.destination_offset,
      byte_count: request.byte_count,
    })
  }
}

///|
/// Copy requested source ranges from inspected shards directly
/// into final caller-owned spans, and visit authority-free metadata.
///
/// The caller must discard destination contents if this function raises.
pub fn copy_ranges(
  root : @approved_fs.ApprovedRoot,
  inspection : StreamingSafetensorsInspection,
  destination~ : FixedArray[Byte],
  requests~ : ArrayView[StreamingSafetensorsCopyRequest],
  limits~ : StreamingSafetensorsLimits,
  visit~ : (
    StreamingSafetensorsTensorMetadata,
    StreamingSafetensorsVisitedRange,
  ) -> Unit raise StreamingSafetensorsError,
) -> Unit raise StreamingSafetensorsError {
  validate_copy_requests(inspection, destination, requests, limits)
  for shard_ordinal, shard in inspection.shards {
    if !requests.any(request => request.shard_ordinal == shard_ordinal) {
      continue
    }
    with_file(root, shard.locator, shard_ordinal, file => {
      copy_open_shard(shard_ordinal, file, shard, destination, requests, visit)
    })
  }
}

///|
fn validate_segmented_copy_requests(
  inspection : StreamingSafetensorsInspection,
  destination_arenas : ArrayView[FixedArray[Byte]],
  requests : ArrayView[StreamingSafetensorsSegmentedCopyRequest],
  limits : StreamingSafetensorsLimits,
) -> Unit raise StreamingSafetensorsError {
  if requests.length() > limits.maximum_copy_requests {
    raise CopyBudgetExceeded
  }
  let mut total = 0L
  for index, request in requests {
    guard inspection.shards.get(request.shard_ordinal) is Some(shard) else {
      raise InvalidCopyRequest(index)
    }
    guard shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) else {
      raise InvalidCopyRequest(index)
    }
    guard destination_arenas.get(request.destination_arena_ordinal)
      is Some(destination) else {
      raise InvalidDestinationArena(index)
    }
    let count = request.byte_count.to_int64()
    if request.tensor_byte_offset > tensor.byte_count() - count {
      raise CopyRangeOutsideTensor(index)
    }
    if request.destination_offset > destination.length() - request.byte_count {
      raise CopyRangeOutsideDestination(index)
    }
    if total > limits.maximum_copy_bytes - count {
      raise CopyBudgetExceeded
    }
    total += count
    for earlier in 0..<index {
      let previous = requests[earlier]
      if previous.destination_arena_ordinal == request.destination_arena_ordinal &&
        previous.destination_offset <
        request.destination_offset + request.byte_count &&
        request.destination_offset <
        previous.destination_offset + previous.byte_count {
        raise OverlappingDestinationRanges(earlier, index)
      }
    }
  }
}

///|
fn copy_open_shard_segmented(
  shard_ordinal : Int,
  file : @approved_fs.ApprovedFile,
  shard : InspectedShard,
  destination_arenas : ArrayView[FixedArray[Byte]],
  requests : ArrayView[StreamingSafetensorsSegmentedCopyRequest],
  visit : (
    StreamingSafetensorsTensorMetadata,
    StreamingSafetensorsSegmentedVisitedRange,
  ) -> Unit raise StreamingSafetensorsError,
) -> Unit raise StreamingSafetensorsError {
  let initial_stamp = stamp_file(shard_ordinal, file)
  if initial_stamp != shard.stamp ||
    initial_stamp.byte_count() != shard.metadata.file_byte_count {
    raise InspectionReplayMismatch(shard_ordinal)
  }
  for request in requests {
    if request.shard_ordinal != shard_ordinal {
      continue
    }
    guard shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) &&
      destination_arenas.get(request.destination_arena_ordinal)
      is Some(destination) else {
      raise InvalidCopyRequest(-1)
    }
    let source_position = shard.metadata.payload_offset +
      tensor.data_start +
      request.tensor_byte_offset
    read_exactly(
      shard_ordinal,
      file,
      destination,
      request.destination_offset,
      request.byte_count,
      source_position,
    )
  }
  if stamp_file(shard_ordinal, file) != initial_stamp {
    raise FileChanged(shard_ordinal)
  }
  for request in requests {
    if request.shard_ordinal != shard_ordinal {
      continue
    }
    guard shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) else {
      raise InvalidCopyRequest(-1)
    }
    visit(tensor, {
      shard_ordinal,
      tensor_ordinal: request.tensor_ordinal,
      tensor_byte_offset: request.tensor_byte_offset,
      destination_arena_ordinal: request.destination_arena_ordinal,
      destination_offset: request.destination_offset,
      byte_count: request.byte_count,
    })
  }
}

///|
/// Copy inspected ranges directly into
/// bounded caller-owned final arenas. The visitor receives metadata only.
///
/// The caller must discard every partially written destination arena if this
/// raises. Payload scrubbing is not required; failed arenas cannot be reused.
pub fn copy_segmented_ranges(
  root : @approved_fs.ApprovedRoot,
  inspection : StreamingSafetensorsInspection,
  destination_arenas~ : ArrayView[FixedArray[Byte]],
  requests~ : ArrayView[StreamingSafetensorsSegmentedCopyRequest],
  limits~ : StreamingSafetensorsLimits,
  visit~ : (
    StreamingSafetensorsTensorMetadata,
    StreamingSafetensorsSegmentedVisitedRange,
  ) -> Unit raise StreamingSafetensorsError,
) -> Unit raise StreamingSafetensorsError {
  validate_segmented_copy_requests(
    inspection, destination_arenas, requests, limits,
  )
  for shard_ordinal, shard in inspection.shards {
    if !requests.any(request => request.shard_ordinal == shard_ordinal) {
      continue
    }
    with_file(root, shard.locator, shard_ordinal, file => {
      copy_open_shard_segmented(
        shard_ordinal, file, shard, destination_arenas, requests, visit,
      )
    })
  }
}
