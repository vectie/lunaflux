///|
fn[S] advance_sharded_transfer(
  prepared : PreparedShardedStreamingLoad[S],
  file_bytes : Int64,
  cursor : ShardedTransferCursor,
) -> Bool raise DeviceWeightError {
  while cursor.order_index < prepared.source_order.length() {
    let reference = prepared.source_order[cursor.order_index]
    let tensor_reference = try! @plan.TensorRef::new(reference)
    guard prepared.rank_plan.tensor(tensor_reference) is Some(placement) else {
      raise InvalidShardedTransfer(reference~, segment=cursor.segment.to_int())
    }
    let recipe = placement.transfer_recipe()
    let count = recipe.segment_count()
    if cursor.segment < count {
      let segment = cursor.segment
      let source_relative = recipe.source_payload_offset_bytes_at(segment) catch {
        _ => raise InvalidShardedTransfer(reference~, segment=segment.to_int())
      }
      let source = sharded_checked_add(
        prepared.payload_offset_bytes,
        source_relative,
        reference,
        segment.to_int(),
      )
      let byte_length = recipe.segment_byte_length()
      let source_end = sharded_checked_add(
        source,
        byte_length,
        reference,
        segment.to_int(),
      )
      let destination = recipe.destination_offset_bytes_at(segment) catch {
        _ => raise InvalidShardedTransfer(reference~, segment=segment.to_int())
      }
      let destination_end = sharded_checked_add(
        destination,
        byte_length,
        reference,
        segment.to_int(),
      )
      if source_end > file_bytes ||
        destination_end > prepared.rank_plan.arena_bytes() {
        raise InvalidShardedTransfer(reference~, segment=segment.to_int())
      }
      cursor.reference = reference
      cursor.source_offset_bytes = source
      cursor.destination_offset_bytes = destination
      cursor.byte_length = byte_length
      cursor.segment = cursor.segment + 1L
      return true
    }
    cursor.order_index = cursor.order_index + 1
    cursor.segment = 0L
  }
  false
}

///|
fn[S : Eq, T] transfer_sharded_streaming_load(
  prepared : PreparedShardedStreamingLoad[S],
  file_bytes : Int64,
  expected_digest : String,
  chunk_bytes : Int,
  resource : T,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise DeviceWeightError,
  observe_stamp : () -> S raise DeviceWeightError,
  copy : (T, FixedArray[Byte], Int, Int64, Int64) -> Unit raise @device.DeviceError,
) -> Unit raise DeviceWeightError {
  let digest = @crypto.SHA256::new()
  let cursor : ShardedTransferCursor = {
    order_index: 0,
    segment: 0L,
    reference: 0,
    source_offset_bytes: 0L,
    destination_offset_bytes: 0L,
    byte_length: 0L,
  }
  let mut has_next = advance_sharded_transfer(prepared, file_bytes, cursor)
  let mut position = 0L
  while position < file_bytes {
    let remaining = file_bytes - position
    let count = if remaining < chunk_bytes.to_int64() {
      remaining.to_int()
    } else {
      chunk_bytes
    }
    read_exact_chunk(read, prepared.scratch, position, count)
    digest.update_from_iter(prepared.scratch[:count].iter())
    let chunk_end = position + count.to_int64()
    let mut scan = true
    while scan {
      if !has_next || cursor.source_offset_bytes >= chunk_end {
        scan = false
      } else {
        let segment_end = cursor.source_offset_bytes + cursor.byte_length
        let overlap_start = cursor.source_offset_bytes.max(position)
        let overlap_end = segment_end.min(chunk_end)
        if overlap_start < overlap_end {
          copy(
            resource,
            prepared.scratch,
            (overlap_start - position).to_int(),
            cursor.destination_offset_bytes +
            (overlap_start - cursor.source_offset_bytes),
            overlap_end - overlap_start,
          ) catch {
            error => raise DeviceFailed(error)
          }
        }
        if segment_end <= chunk_end {
          has_next = advance_sharded_transfer(prepared, file_bytes, cursor)
        } else {
          scan = false
        }
      }
    }
    position = chunk_end
  }
  if has_next {
    raise InvalidShardedTransfer(
      reference=cursor.reference,
      segment=(cursor.segment - 1L).to_int(),
    )
  }
  let observed_stamp = observe_stamp() catch {
    _ => raise FileChangedDuringLoad
  }
  if observed_stamp != prepared.source_stamp ||
    @crypto.bytes_to_hex_string(digest.finalize()) != expected_digest {
    raise FileChangedDuringLoad
  }
}

///|
fn[T] sharded_resource_failure(
  resource : T,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  cause : DeviceWeightError,
  stage : DeviceWeightFailureStage,
  close : (T) -> Unit raise @device.DeviceError,
) -> ShardedResourcePreparation[T] raise DeviceWeightError {
  try close(resource) catch {
    cleanup_failure =>
      ShardedResourceCleanupRequired({
        resource,
        rank_plan,
        cause,
        stage,
        cleanup_failure,
      })
  } noraise {
    _ => raise cause
  }
}

///|
fn[S : Eq, T] load_prepared_sharded_streaming_with_resource(
  prepared : PreparedShardedStreamingLoad[S],
  file_bytes : Int64,
  expected_digest : String,
  chunk_bytes : Int,
  read : (FixedArray[Byte], Int, Int64, Int) -> Unit raise DeviceWeightError,
  observe_stamp : () -> S raise DeviceWeightError,
  open : (Int64) -> T raise @device.DeviceError,
  copy : (T, FixedArray[Byte], Int, Int64, Int64) -> Unit raise @device.DeviceError,
  close : (T) -> Unit raise @device.DeviceError,
) -> ShardedResourcePreparation[T] raise DeviceWeightError {
  let resource = open(prepared.rank_plan.arena_bytes()) catch {
    error => raise DeviceFailed(error)
  }
  let transferred : Result[Unit, DeviceWeightError] = try {
    transfer_sharded_streaming_load(
      prepared, file_bytes, expected_digest, chunk_bytes, resource, read, observe_stamp,
      copy,
    )
    Ok(())
  } catch {
    error => Err(error)
  }
  match transferred {
    Ok(_) => ShardedResourceReady({ resource, rank_plan: prepared.rank_plan })
    Err(error) => {
      let stage = match error {
        DeviceFailed(_) => Transfer
        _ => SourceRevalidation
      }
      sharded_resource_failure(
        resource,
        prepared.rank_plan,
        error,
        stage,
        close,
      )
    }
  }
}
