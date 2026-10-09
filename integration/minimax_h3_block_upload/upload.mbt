///|
pub(all) suberror BlockUploadError {
  InvalidChunkSize
  TransferFailed
}

///|
fn copy_chunks(
  arena : FixedArray[Byte],
  source : Int64,
  destination : Int64,
  length : Int64,
  chunk_bytes : Int,
  consume : (Bytes, Int64) -> Bool,
) -> Bool {
  if chunk_bytes <= 0 ||
    source < 0L ||
    length <= 0L ||
    source > arena.length().to_int64() ||
    length > arena.length().to_int64() - source {
    return false
  }
  for copied = 0L; copied < length; {
    let count = Int64::min(chunk_bytes.to_int64(), length - copied).to_int()
    let start = (source + copied).to_int()
    let bytes = Bytes::from_array(arena[start:start + count])
    if !consume(bytes, destination + copied) {
      return false
    }
    continue copied + count.to_int64()
  }
  true
}

///|
/// Upload a packed block from an existing host component into caller-owned
/// device storage. Startup only. Failed transfer leaves the destination partial
/// and owned by the caller: do not execute it; retry this entire upload.
pub fn upload_block(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @minimax_h3_weights.BlockWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
  destination_offset~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_block_weights(packing, expected_identity~)
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    packing.byte_count,
    destination_offset~,
    chunk_bytes~,
  )
}

///|
pub fn upload_output(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  packing : @minimax_h3_weights.OutputWeightPacking,
  expected_identity~ : @spec.ModelIdentity,
  destination_offset~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard chunk_bytes > 0 else { raise InvalidChunkSize }
  let copies = host.resolve_output_weights(packing, expected_identity~)
  upload_resolved(
    context,
    destination,
    host,
    copies[:],
    packing.byte_count,
    destination_offset~,
    chunk_bytes~,
  )
}

///|
fn upload_resolved(
  context : @device.Context,
  destination : @device.Allocation,
  host : @minimax_h3_host_materialization.MiniMaxH3HostComponentArena,
  copies : ArrayView[@minimax_h3_host_materialization.ResolvedBlockWeightCopy],
  byte_count : Int64,
  destination_offset~ : Int64,
  chunk_bytes~ : Int,
) -> Unit raise {
  context.validate_allocation_region(
    destination,
    offset=destination_offset,
    byte_count~,
    alignment=2L,
  )
  for copy in copies {
    let consumer = @segmented_host_borrow.SegmentedHostArenaConsumer::new(
      arena_ordinal=copy.arena_index,
      expected_length=host.device_arena_size(copy.arena_index),
      consume=arena => {
        copy_chunks(
          arena,
          copy.source_offset,
          destination_offset + copy.destination_offset,
          copy.byte_count,
          chunk_bytes,
          (bytes, offset) => {
            try
              destination.copy_from_host(
                bytes,
                0L,
                offset,
                bytes.length().to_int64(),
              )
            catch {
              _ => false
            } noraise {
              _ => true
            }
          },
        )
      },
    )
    host.provide_device_arena(copy.arena_index, consumer) catch {
      _ => {
        consumer.revoke()
        raise TransferFailed
      }
    }
    consumer.revoke()
  }
}
