///|
fn expected_peer_link_count(world_size : Int) -> Int {
  world_size * (world_size - 1) / 2
}

///|
fn facts_from_topology(
  topology : @device_topology.AdmittedLocalTopology,
) -> TopologyFacts raise TensorParallelAdmissionError {
  let world_size = topology.world_size()
  let ranks : Array[TopologyRankFacts] = []
  for rank in 0..<world_size {
    guard topology.rank(rank) is Some(value) else {
      raise reject(Topology, CountMismatch, rank)
    }
    ranks.push({
      rank,
      process_ordinal: value.ordinal(),
      device_name: value.name(),
      physical_bytes: value.total_memory_bytes(),
      target: value.target(),
    })
  }
  {
    driver_version: topology.driver_version(),
    world_size,
    target: topology.target(),
    peer_link_count: topology.link_count(),
    ranks: ReadOnlyArray::from_array(ranks),
  }
}

///|
fn validate_topology_facts(
  facts : TopologyFacts,
) -> Unit raise TensorParallelAdmissionError {
  if facts.driver_version <= 0 ||
    facts.world_size < MIN_TENSOR_PARALLEL_RANKS ||
    facts.world_size > MAX_TENSOR_PARALLEL_RANKS ||
    facts.ranks.length() != facts.world_size {
    raise reject(Topology, CountMismatch, -1)
  }
  if facts.peer_link_count != expected_peer_link_count(facts.world_size) {
    raise reject(Topology, IncompletePeerLinks, -1)
  }
  for rank, value in facts.ranks {
    if value.rank != rank {
      raise reject(Topology, RankMismatch, rank)
    }
    if value.process_ordinal < 0 ||
      value.process_ordinal > 1023 ||
      !valid_device_name(value.device_name) ||
      value.physical_bytes <= 0L {
      raise reject(Topology, DeviceIdentityMismatch, rank)
    }
    if value.target != facts.target {
      raise reject(Topology, TargetMismatch, rank)
    }
  }
}

///|
fn admit_rank(
  facts : TopologyFacts,
  rank : Int,
  claim : TensorParallelRankDeclaration,
) -> TensorParallelRankCapacity raise TensorParallelAdmissionError {
  let actual = facts.ranks[rank]
  if claim.rank != rank {
    raise reject(RankBinding, RankMismatch, rank)
  }
  if claim.world_size != facts.world_size {
    raise reject(RankBinding, WorldMismatch, rank)
  }
  if claim.process_ordinal != actual.process_ordinal {
    raise reject(RankBinding, OrdinalMismatch, rank)
  }
  if claim.device_name != actual.device_name {
    raise reject(RankBinding, DeviceIdentityMismatch, rank)
  }
  if claim.target != facts.target || claim.target != actual.target {
    raise reject(RankBinding, TargetMismatch, rank)
  }
  if claim.peer_rank_count != facts.world_size - 1 {
    raise reject(RankBinding, IncompletePeerLinks, rank)
  }
  if claim.memory_ceiling_bytes < MIN_RANK_MEMORY_CEILING_BYTES ||
    claim.memory_ceiling_bytes > MAX_RANK_MEMORY_CEILING_BYTES {
    raise reject(RankMemory, InvalidMemory, rank)
  }
  if claim.memory_ceiling_bytes > actual.physical_bytes {
    raise reject(RankMemory, CeilingExceedsPhysical, rank)
  }
  let reserved_total_bytes = reservation_total(
    claim.weight_reserved_bytes,
    claim.activation_workspace_reserved_bytes,
    claim.kv_reserved_bytes,
    claim.collective_reserved_bytes,
    rank,
  )
  if reserved_total_bytes != claim.reserved_total_bytes ||
    reserved_total_bytes > claim.memory_ceiling_bytes {
    raise reject(RankMemory, ReservationExceedsCeiling, rank)
  }
  {
    rank,
    world_size: facts.world_size,
    process_ordinal: actual.process_ordinal,
    device_name: actual.device_name,
    target: actual.target,
    peer_rank_count: claim.peer_rank_count,
    physical_bytes: actual.physical_bytes,
    memory_ceiling_bytes: claim.memory_ceiling_bytes,
    weight_reserved_bytes: claim.weight_reserved_bytes,
    activation_workspace_reserved_bytes: claim.activation_workspace_reserved_bytes,
    kv_reserved_bytes: claim.kv_reserved_bytes,
    collective_reserved_bytes: claim.collective_reserved_bytes,
    reserved_total_bytes,
    available_bytes: claim.memory_ceiling_bytes - reserved_total_bytes,
  }
}

///|
fn add_rank_to_aggregate(
  aggregate : TensorParallelAggregateCapacity,
  rank : TensorParallelRankCapacity,
) -> TensorParallelAggregateCapacity raise TensorParallelAdmissionError {
  {
    physical_bytes: checked_nonnegative_add(
      aggregate.physical_bytes,
      rank.physical_bytes,
      Aggregate,
      -1,
    ),
    memory_ceiling_bytes: checked_nonnegative_add(
      aggregate.memory_ceiling_bytes,
      rank.memory_ceiling_bytes,
      Aggregate,
      -1,
    ),
    weight_reserved_bytes: checked_nonnegative_add(
      aggregate.weight_reserved_bytes,
      rank.weight_reserved_bytes,
      Aggregate,
      -1,
    ),
    activation_workspace_reserved_bytes: checked_nonnegative_add(
      aggregate.activation_workspace_reserved_bytes,
      rank.activation_workspace_reserved_bytes,
      Aggregate,
      -1,
    ),
    kv_reserved_bytes: checked_nonnegative_add(
      aggregate.kv_reserved_bytes,
      rank.kv_reserved_bytes,
      Aggregate,
      -1,
    ),
    collective_reserved_bytes: checked_nonnegative_add(
      aggregate.collective_reserved_bytes,
      rank.collective_reserved_bytes,
      Aggregate,
      -1,
    ),
    reserved_total_bytes: checked_nonnegative_add(
      aggregate.reserved_total_bytes,
      rank.reserved_total_bytes,
      Aggregate,
      -1,
    ),
    available_bytes: checked_nonnegative_add(
      aggregate.available_bytes,
      rank.available_bytes,
      Aggregate,
      -1,
    ),
    minimum_rank_available_bytes: if aggregate.minimum_rank_available_bytes <
      rank.available_bytes {
      aggregate.minimum_rank_available_bytes
    } else {
      rank.available_bytes
    },
  }
}

///|
fn admit_facts(
  facts : TopologyFacts,
  declaration : TensorParallelRuntimeDeclaration,
) -> TensorParallelRuntimeAdmission raise TensorParallelAdmissionError {
  validate_topology_facts(facts)
  if declaration.host_count != 1 {
    raise reject(Support, CrossNodeUnsupported, -1)
  }
  if declaration.pipeline_stage_count != 1 {
    raise reject(Support, PipelineUnsupported, -1)
  }
  if declaration.ranks.length() != facts.world_size {
    raise reject(RankBinding, CountMismatch, -1)
  }
  let ranks : Array[TensorParallelRankCapacity] = []
  let mut aggregate : TensorParallelAggregateCapacity = {
    physical_bytes: 0L,
    memory_ceiling_bytes: 0L,
    weight_reserved_bytes: 0L,
    activation_workspace_reserved_bytes: 0L,
    kv_reserved_bytes: 0L,
    collective_reserved_bytes: 0L,
    reserved_total_bytes: 0L,
    available_bytes: 0L,
    minimum_rank_available_bytes: MAX_INT64,
  }
  for rank, claim in declaration.ranks {
    let admitted = admit_rank(facts, rank, claim)
    aggregate = add_rank_to_aggregate(aggregate, admitted)
    ranks.push(admitted)
  }
  let category_total = reservation_total(
    aggregate.weight_reserved_bytes,
    aggregate.activation_workspace_reserved_bytes,
    aggregate.kv_reserved_bytes,
    aggregate.collective_reserved_bytes,
    -1,
  )
  if category_total != aggregate.reserved_total_bytes ||
    aggregate.reserved_total_bytes > aggregate.memory_ceiling_bytes ||
    aggregate.available_bytes !=
    aggregate.memory_ceiling_bytes - aggregate.reserved_total_bytes {
    raise reject(Aggregate, Overflow, -1)
  }
  {
    driver_version: facts.driver_version,
    world_size: facts.world_size,
    target: facts.target,
    peer_link_count: facts.peer_link_count,
    ranks: ReadOnlyArray::from_array(ranks),
    aggregate,
  }
}

///|
/// Authenticates one explicit same-host tensor-parallel deployment declaration
/// against already admitted physical topology evidence.
pub fn admit(
  topology : @device_topology.AdmittedLocalTopology,
  declaration : TensorParallelRuntimeDeclaration,
) -> TensorParallelRuntimeAdmission raise TensorParallelAdmissionError {
  admit_facts(facts_from_topology(topology), declaration)
}
