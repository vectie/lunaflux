///|
/// A tensor slice destined for an opaque caller-owned storage region. Unlike a
/// managed-array copy, both its length and final offset can exceed signed Int.
pub struct StreamingSafetensorsTransfer {
  priv shard_ordinal : Int
  priv tensor_ordinal : Int
  priv tensor_byte_offset : Int64
  priv destination_ordinal : Int
  priv destination_offset : Int64
  priv byte_count : Int64
}

///|
pub fn StreamingSafetensorsTransfer::new(
  shard_ordinal~ : Int,
  tensor_ordinal~ : Int,
  tensor_byte_offset~ : Int64,
  destination_ordinal~ : Int,
  destination_offset~ : Int64,
  byte_count~ : Int64,
) -> Self raise StreamingSafetensorsError {
  guard shard_ordinal >= 0 &&
    tensor_ordinal >= 0 &&
    tensor_byte_offset >= 0L &&
    destination_ordinal >= 0 &&
    destination_offset >= 0L &&
    byte_count > 0L &&
    byte_count <= 9223372036854775807L - destination_offset else {
    raise InvalidCopyRequest(-1)
  }
  {
    shard_ordinal,
    tensor_ordinal,
    tensor_byte_offset,
    destination_ordinal,
    destination_offset,
    byte_count,
  }
}

///|
fn validate_transfers(
  inspection : StreamingSafetensorsInspection,
  requests : ArrayView[StreamingSafetensorsTransfer],
  destination_bytes : ArrayView[Int64],
  limits : StreamingSafetensorsLimits,
) -> Unit raise StreamingSafetensorsError {
  guard requests.length() <= limits.maximum_copy_requests else {
    raise CopyBudgetExceeded
  }
  let mut total = 0L
  for i, request in requests {
    guard inspection.shards.get(request.shard_ordinal) is Some(shard) &&
      shard.metadata.tensors.get(request.tensor_ordinal) is Some(tensor) else {
      raise InvalidCopyRequest(i)
    }
    guard destination_bytes.get(request.destination_ordinal) is Some(bytes) else {
      raise InvalidDestinationArena(i)
    }
    guard request.byte_count <= tensor.byte_count() &&
      request.tensor_byte_offset <= tensor.byte_count() - request.byte_count else {
      raise CopyRangeOutsideTensor(i)
    }
    guard bytes >= request.byte_count &&
      request.destination_offset <= bytes - request.byte_count else {
      raise CopyRangeOutsideDestination(i)
    }
    guard request.byte_count <= limits.maximum_copy_bytes - total else {
      raise CopyBudgetExceeded
    }
    total += request.byte_count
    for earlier in 0..<i {
      let previous = requests[earlier]
      if previous.destination_ordinal == request.destination_ordinal &&
        previous.destination_offset <
        request.destination_offset + request.byte_count &&
        request.destination_offset <
        previous.destination_offset + previous.byte_count {
        raise OverlappingDestinationRanges(earlier, i)
      }
    }
  }
}

///|
/// Stream authenticated slices through one reusable caller-owned scratch span.
/// `consume(destination, offset, scratch, count)` must finish consuming the
/// first `count` bytes synchronously: the next chunk overwrites them. Returning
/// false stops immediately. Destinations stay caller-owned, and must not be
/// executed after failure. No complete tensor or host weight arena is built.
/// Each referenced shard is opened and reauthenticated once, not once per chunk.
pub fn transfer_ranges(
  root : @approved_fs.ApprovedRoot,
  inspection : StreamingSafetensorsInspection,
  requests~ : ArrayView[StreamingSafetensorsTransfer],
  destination_bytes~ : ArrayView[Int64],
  scratch~ : FixedArray[Byte],
  limits~ : StreamingSafetensorsLimits,
  consume~ : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise StreamingSafetensorsError {
  guard scratch.length() > 0 else { raise InvalidLimit }
  validate_transfers(inspection, requests, destination_bytes, limits)
  for shard_ordinal, shard in inspection.shards {
    if !requests.any(request => request.shard_ordinal == shard_ordinal) {
      continue
    }
    with_file(root, shard.locator, shard_ordinal, file => {
      let stamp = stamp_file(shard_ordinal, file)
      guard stamp == shard.stamp else {
        raise InspectionReplayMismatch(shard_ordinal)
      }
      authenticate_file(
        shard_ordinal,
        file,
        shard.metadata.file_byte_count,
        shard.expected_digest,
        limits.authentication_chunk_bytes,
      )
      guard stamp_file(shard_ordinal, file) == stamp else {
        raise FileChanged(shard_ordinal)
      }
      for request_index, request in requests {
        if request.shard_ordinal != shard_ordinal {
          continue
        }
        let tensor = shard.metadata.tensors[request.tensor_ordinal]
        let start = shard.metadata.payload_offset +
          tensor.data_start +
          request.tensor_byte_offset
        let mut copied = 0L
        while copied < request.byte_count {
          let count = Int64::min(
            scratch.length().to_int64(),
            request.byte_count - copied,
          ).to_int()
          read_exactly(shard_ordinal, file, scratch, 0, count, start + copied)
          if !consume(
              request.destination_ordinal,
              request.destination_offset + copied,
              scratch,
              count,
            ) {
            raise VisitorRejected(request_index)
          }
          copied += count.to_int64()
        }
      }
      guard stamp_file(shard_ordinal, file) == stamp else {
        raise FileChanged(shard_ordinal)
      }
    })
  }
}
