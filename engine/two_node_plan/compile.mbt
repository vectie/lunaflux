///|
fn checked_add(
  a : Int64,
  b : Int64,
  rank : Int,
) -> Int64 raise TwoNodePlanError {
  if a < 0L || b < 0L || a > 9223372036854775807L - b {
    raise Rejected(rule=Overflow, rank~)
  }
  a + b
}

///|
fn checked_multiply(a : Int64, b : Int64) -> Int64 raise TwoNodePlanError {
  if a <= 0L || b <= 0L || a > 9223372036854775807L / b {
    raise Rejected(rule=Overflow, rank=-1)
  }
  a * b
}

///|
fn local_kv_layout(
  model : @plan.ModelPlan,
  tokens_per_page : Int,
  page_count : Int,
) -> @device_layout.DeviceKvLayout raise TwoNodePlanError {
  let geometry = model.kv_cache_geometry()
  if tokens_per_page <= 0 || geometry.kv_head_count() % 2 != 0 {
    raise Rejected(rule=KvGeometry, rank=-1)
  }
  if geometry.dtype() != BFloat16 {
    raise Rejected(rule=NumericFormat, rank=-1)
  }
  let local_geometry = @plan.KvCacheGeometry::new(
    layer_count=geometry.layer_count(),
    kv_head_count=geometry.kv_head_count() / 2,
    head_dimension=geometry.head_dimension(),
    dtype=geometry.dtype(),
    layout=geometry.layout(),
  ) catch {
    _ => raise Rejected(rule=KvGeometry, rank=-1)
  }
  let limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page~,
    total_page_count=page_count,
    max_arena_bytes=9223372036854775807L,
  ) catch {
    _ => raise Rejected(rule=KvGeometry, rank=-1)
  }
  @device_layout.plan(model.identity(), local_geometry, limits) catch {
    SizeOverflow(_) => raise Rejected(rule=Overflow, rank=-1)
    _ => raise Rejected(rule=KvGeometry, rank=-1)
  }
}

///|
fn capacity(
  node : @network_topology.NetworkRank,
  plan : @tensor_parallel_plan.TensorParallelRankPlan,
  budget : RankMemoryBudget,
  page_bytes : Int64,
  rank : Int,
) -> RankCapacity raise TwoNodePlanError {
  if budget.system_reserved_bytes >= node.physical_memory_bytes() ||
    budget.runtime_ceiling_bytes >
    node.physical_memory_bytes() - budget.system_reserved_bytes {
    raise Rejected(rule=MemoryBudget, rank~)
  }
  let reserved = checked_add(
    plan.arena_bytes(),
    budget.activation_workspace_bytes,
    rank,
  )
  let reserved = checked_add(reserved, budget.collective_bytes, rank)
  let reserved = checked_add(reserved, budget.kv_bytes, rank)
  if reserved > budget.runtime_ceiling_bytes || budget.kv_bytes < page_bytes {
    raise Rejected(rule=RankCapacity, rank~)
  }
  {
    weight_bytes: plan.arena_bytes(),
    reserved_bytes: reserved,
    available_bytes: budget.runtime_ceiling_bytes - reserved,
    kv_page_bytes: page_bytes,
    maximum_kv_pages: budget.kv_bytes / page_bytes,
  }
}

///|
/// Pure startup compilation over existing family-produced rank plans.
/// Budgets are declared reservations; artifact/workspace and live admission
/// must independently verify them before execution. No resources are acquired.
pub fn compile(
  model : @plan.ModelPlan,
  topology : @network_topology.TwoNodeTopology,
  ranks : ArrayView[@tensor_parallel_plan.TensorParallelRankPlan],
  budgets : ArrayView[RankMemoryBudget],
  tokens_per_page~ : Int,
  maximum_pages~ : Int,
) -> TwoNodePlan raise TwoNodePlanError {
  if ranks.length() != 2 || budgets.length() != 2 {
    raise Rejected(rule=RankOrder, rank=-1)
  }
  if maximum_pages <= 0 {
    raise Rejected(rule=KvGeometry, rank=-1)
  }
  if !topology.target().supports_bf16() {
    raise Rejected(rule=NumericFormat, rank=-1)
  }
  for rank, plan in ranks {
    if plan.topology().rank() != rank || plan.topology().world_size() != 2 {
      raise Rejected(rule=RankOrder, rank~)
    }
    if plan.identity() != model.identity() {
      raise Rejected(rule=ModelIdentity, rank~)
    }
  }
  validate_pair(model, ranks[0], ranks[1])
  // Canonical planning includes aligned K/V segments, not only element bytes.
  let page_bytes = local_kv_layout(model, tokens_per_page, 1).arena_bytes()
  let capacities : Array[RankCapacity] = []
  for rank in 0..<2 {
    guard topology.rank(rank) is Some(node) else {
      raise Rejected(rule=RankOrder, rank~)
    }
    capacities.push(
      capacity(node, ranks[rank], budgets[rank], page_bytes, rank),
    )
  }
  let pages = Int64::min(
    maximum_pages.to_int64(),
    Int64::min(capacities[0].maximum_kv_pages, capacities[1].maximum_kv_pages),
  )
  let kv_layout = local_kv_layout(model, tokens_per_page, pages.to_int())
  {
    kv_layout,
    digest: digest_plan(model, topology, ranks, budgets, kv_layout),
    model,
    topology,
    rank_plans: ReadOnlyArray::from_array(ranks),
    budgets: ReadOnlyArray::from_array(budgets),
    capacities: ReadOnlyArray::from_array(capacities),
    tokens_per_page,
    page_count: pages.to_int(),
  }
}
