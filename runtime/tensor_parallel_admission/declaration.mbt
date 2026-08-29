///|
fn valid_device_name(value : StringView) -> Bool {
  if value.is_empty() || value.length() > 256 {
    return false
  }
  for character in value {
    if character < ' ' || character == '\u{7f}' {
      return false
    }
  }
  true
}

///|
pub fn TensorParallelRankDeclaration::new(
  rank~ : Int,
  world_size~ : Int,
  process_ordinal~ : Int,
  device_name~ : StringView,
  target~ : @catalog.DeviceTarget,
  peer_rank_count~ : Int,
  memory_ceiling_bytes~ : Int64,
  weight_reserved_bytes~ : Int64,
  activation_workspace_reserved_bytes~ : Int64,
  kv_reserved_bytes~ : Int64,
  collective_reserved_bytes~ : Int64,
) -> TensorParallelRankDeclaration raise TensorParallelAdmissionError {
  if world_size < MIN_TENSOR_PARALLEL_RANKS ||
    world_size > MAX_TENSOR_PARALLEL_RANKS {
    raise reject(Declaration, InvalidWorld, rank)
  }
  if rank < 0 || rank >= world_size {
    raise reject(Declaration, InvalidRank, rank)
  }
  if process_ordinal < 0 || process_ordinal > 1023 {
    raise reject(Declaration, InvalidOrdinal, rank)
  }
  if !valid_device_name(device_name) {
    raise reject(Declaration, InvalidDeviceIdentity, rank)
  }
  if peer_rank_count < 0 || peer_rank_count >= MAX_TENSOR_PARALLEL_RANKS {
    raise reject(Declaration, InvalidPeerCount, rank)
  }
  if memory_ceiling_bytes < MIN_RANK_MEMORY_CEILING_BYTES ||
    memory_ceiling_bytes > MAX_RANK_MEMORY_CEILING_BYTES {
    raise reject(Declaration, InvalidMemory, rank)
  }
  let reserved_total_bytes = reservation_total(
    weight_reserved_bytes, activation_workspace_reserved_bytes, kv_reserved_bytes,
    collective_reserved_bytes, rank,
  )
  if reserved_total_bytes > memory_ceiling_bytes {
    raise reject(RankMemory, ReservationExceedsCeiling, rank)
  }
  {
    rank,
    world_size,
    process_ordinal,
    device_name: device_name.to_owned(),
    target,
    peer_rank_count,
    memory_ceiling_bytes,
    weight_reserved_bytes,
    activation_workspace_reserved_bytes,
    kv_reserved_bytes,
    collective_reserved_bytes,
    reserved_total_bytes,
  }
}

///|
pub fn TensorParallelRuntimeDeclaration::new(
  host_count~ : Int,
  pipeline_stage_count~ : Int,
  ranks : ArrayView[TensorParallelRankDeclaration],
) -> TensorParallelRuntimeDeclaration raise TensorParallelAdmissionError {
  if host_count <= 0 ||
    host_count > MAX_TENSOR_PARALLEL_RANKS ||
    pipeline_stage_count <= 0 ||
    pipeline_stage_count > MAX_TENSOR_PARALLEL_RANKS {
    raise reject(Declaration, InvalidCount, -1)
  }
  if ranks.length() < MIN_TENSOR_PARALLEL_RANKS ||
    ranks.length() > MAX_TENSOR_PARALLEL_RANKS {
    raise reject(Declaration, InvalidCount, -1)
  }
  { host_count, pipeline_stage_count, ranks: ReadOnlyArray::from_array(ranks) }
}
