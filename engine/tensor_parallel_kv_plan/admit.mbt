///|
const MAX_ARENA_BYTES : Int64 = 9223372036854775807L

///|
fn translate_layout_failure(
  body : () -> @device_layout.DeviceKvLayout raise,
  rank : Int,
) -> @device_layout.DeviceKvLayout raise TensorParallelKvPlanError {
  body() catch {
    @device_layout.DeviceKvLayoutError::SizeOverflow(_) =>
      raise rejected(SizeOverflow, rank)
    @device_layout.DeviceKvLayoutError::ArenaLimitExceeded =>
      raise rejected(RankArenaCeiling, rank)
    _ => raise rejected(LayoutVersion, rank)
  }
}

///|
fn admit_facts(
  identity : @spec.ModelIdentity,
  geometry : @plan.KvCacheGeometry,
  generation_value : UInt64,
  devices : ArrayView[RankDeviceFacts],
  runtime_target : @catalog.DeviceTarget,
  runtime_ranks : ArrayView[RankRuntimeFacts],
  tokens_per_page : Int,
  total_page_count : Int,
) -> TensorParallelKvPlan raise TensorParallelKvPlanError {
  let world_size = devices.length()
  if world_size <= 0 || runtime_ranks.length() != world_size {
    raise rejected(BindingCount, -1)
  }
  if generation_value == 0UL {
    raise rejected(ModelGeneration, -1)
  }
  let first_target = devices[0].target
  let ceilings : Array[Int64] = []
  for rank in 0..<world_size {
    let device = devices[rank]
    let runtime = runtime_ranks[rank]
    if device.rank != rank || runtime.rank != rank {
      raise rejected(Rank, rank)
    }
    if device.world_size != world_size || runtime.world_size != world_size {
      raise rejected(WorldSize, rank)
    }
    if device.identity != identity {
      raise rejected(ModelIdentity, rank)
    }
    if device.target != first_target ||
      device.target != runtime_target ||
      runtime.target != runtime_target {
      raise rejected(DeviceTarget, rank)
    }
    if device.device_ordinal != runtime.process_ordinal {
      raise rejected(DeviceOrdinal, rank)
    }
    if runtime.kv_reserved_bytes <= 0L {
      raise rejected(RankArenaCeiling, rank)
    }
    ceilings.push(runtime.kv_reserved_bytes)
  }
  let global_heads = geometry.kv_head_count()
  if global_heads % world_size != 0 {
    raise rejected(KvHeadDivisibility, -1)
  }
  let local_heads = global_heads / world_size
  let local_geometry = @plan.KvCacheGeometry::new(
    layer_count=geometry.layer_count(),
    kv_head_count=local_heads,
    head_dimension=geometry.head_dimension(),
    dtype=geometry.dtype(),
    layout=geometry.layout(),
  ) catch {
    _ => raise rejected(KvHeadDivisibility, -1)
  }
  let local_limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page~,
    total_page_count~,
    max_arena_bytes=MAX_ARENA_BYTES,
  ) catch {
    _ => raise rejected(SizeOverflow, -1)
  }
  let local_layout = translate_layout_failure(
    () => @device_layout.plan(identity, local_geometry, local_limits),
    -1,
  )
  let full_limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page~,
    total_page_count~,
    max_arena_bytes=MAX_ARENA_BYTES,
  ) catch {
    _ => raise rejected(SizeOverflow, -1)
  }
  let full_layout = translate_layout_failure(
    () => @device_layout.plan(identity, geometry, full_limits),
    -1,
  )
  if local_layout.page_logical_bytes() !=
    full_layout.page_logical_bytes() / Int64::from_int(world_size) ||
    full_layout.page_logical_bytes() % Int64::from_int(world_size) != 0L {
    raise rejected(SizeOverflow, -1)
  }
  if world_size > 1 && local_layout.arena_bytes() >= full_layout.arena_bytes() {
    raise rejected(FullArenaReplication, -1)
  }
  for rank in 0..<world_size {
    if local_layout.arena_bytes() > ceilings[rank] {
      raise rejected(RankArenaCeiling, rank)
    }
  }
  let rank_bindings : Array[TensorParallelKvRankBinding] = []
  for rank in 0..<world_size {
    rank_bindings.push(
      build_rank_binding(
        identity,
        generation_value,
        rank,
        world_size,
        devices[rank].device_ordinal,
        devices[rank].target,
        global_heads,
        local_layout,
        ceilings[rank],
      ),
    )
  }
  {
    identity,
    generation_value,
    world_size,
    global_kv_head_count: global_heads,
    local_kv_head_count: local_heads,
    local_layout,
    full_arena_bytes: full_layout.arena_bytes(),
    rank_kv_reservations: ReadOnlyArray::from_array(ceilings),
    rank_bindings: ReadOnlyArray::from_array(rank_bindings),
  }
}

///|
/// Authenticates a rank-ordered device-plan set and derives the immutable
/// rank-local KV arena geometry used by every rank.
pub fn admit(
  model_plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  runtime : @tensor_parallel_admission.TensorParallelRuntimeAdmission,
  devices : ArrayView[@tensor_parallel_device_plan.TensorParallelDevicePlan],
  tokens_per_page~ : Int,
  total_page_count~ : Int,
) -> TensorParallelKvPlan raise TensorParallelKvPlanError {
  let facts : Array[RankDeviceFacts] = []
  for device in devices {
    facts.push({
      identity: device.identity(),
      rank: device.rank(),
      world_size: device.world_size(),
      device_ordinal: device.device_ordinal(),
      target: device.target(),
    })
  }
  let runtime_facts : Array[RankRuntimeFacts] = []
  for rank in 0..<runtime.world_size() {
    guard runtime.rank_capacity(rank) is Some(capacity) else {
      raise rejected(BindingCount, rank)
    }
    runtime_facts.push({
      rank: capacity.rank(),
      world_size: capacity.world_size(),
      process_ordinal: capacity.process_ordinal(),
      target: capacity.target(),
      kv_reserved_bytes: capacity.kv_reserved_bytes(),
    })
  }
  admit_facts(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    generation.value(),
    facts,
    runtime.target(),
    runtime_facts,
    tokens_per_page,
    total_page_count,
  )
}
