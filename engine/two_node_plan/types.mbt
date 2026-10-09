///|
pub(all) enum TwoNodePlanRule {
  RankOrder
  ModelIdentity
  TensorPartition
  OperationMismatch
  CollectiveMismatch
  NumericFormat
  MemoryBudget
  RankCapacity
  KvGeometry
  Overflow
} derive(Debug, Eq)

///|
pub(all) suberror TwoNodePlanError {
  Rejected(rule~ : TwoNodePlanRule, rank~ : Int)
} derive(Debug, Eq)

///|
/// Startup memory inputs for one node. Weight bytes come from the rank plan.
pub struct RankMemoryBudget {
  priv system_reserved_bytes : Int64
  priv runtime_ceiling_bytes : Int64
  priv activation_workspace_bytes : Int64
  priv collective_bytes : Int64
  priv kv_bytes : Int64
} derive(Debug, Eq)

///|
pub fn RankMemoryBudget::new(
  system_reserved_bytes~ : Int64,
  runtime_ceiling_bytes~ : Int64,
  activation_workspace_bytes~ : Int64,
  collective_bytes~ : Int64,
  kv_bytes~ : Int64,
) -> RankMemoryBudget raise TwoNodePlanError {
  if system_reserved_bytes <= 0L ||
    runtime_ceiling_bytes <= 0L ||
    activation_workspace_bytes < 0L ||
    collective_bytes < 0L ||
    kv_bytes <= 0L {
    raise Rejected(rule=MemoryBudget, rank=-1)
  }
  {
    system_reserved_bytes,
    runtime_ceiling_bytes,
    activation_workspace_bytes,
    collective_bytes,
    kv_bytes,
  }
}

///|
pub struct RankCapacity {
  priv weight_bytes : Int64
  priv reserved_bytes : Int64
  priv available_bytes : Int64
  priv kv_page_bytes : Int64
  priv maximum_kv_pages : Int64
} derive(Debug, Eq)

///|
pub fn RankCapacity::weight_bytes(self : RankCapacity) -> Int64 {
  self.weight_bytes
}

///|
pub fn RankCapacity::reserved_bytes(self : RankCapacity) -> Int64 {
  self.reserved_bytes
}

///|
pub fn RankCapacity::available_bytes(self : RankCapacity) -> Int64 {
  self.available_bytes
}

///|
pub fn RankCapacity::kv_page_bytes(self : RankCapacity) -> Int64 {
  self.kv_page_bytes
}

///|
pub fn RankCapacity::maximum_kv_pages(self : RankCapacity) -> Int64 {
  self.maximum_kv_pages
}

///|
/// Inert compiler output. Never grants worker, device, or network authority.
pub struct TwoNodePlan {
  priv digest : TwoNodePlanDigest
  priv kv_layout : @device_layout.DeviceKvLayout
  priv model : @plan.ModelPlan
  priv topology : @network_topology.TwoNodeTopology
  priv rank_plans : ReadOnlyArray[@tensor_parallel_plan.TensorParallelRankPlan]
  priv budgets : ReadOnlyArray[RankMemoryBudget]
  priv capacities : ReadOnlyArray[RankCapacity]
  priv tokens_per_page : Int
  priv page_count : Int
} derive(Debug, Eq)

///|
/// Identity of the complete pure plan, computed once during compilation.
pub struct TwoNodePlanDigest(String) derive(Debug, Eq)

///|
pub fn TwoNodePlanDigest::as_sha256(self : TwoNodePlanDigest) -> String {
  self.0
}

///|
pub fn TwoNodePlan::digest(self : TwoNodePlan) -> TwoNodePlanDigest {
  self.digest
}

///|
pub fn TwoNodePlan::kv_layout(
  self : TwoNodePlan,
) -> @device_layout.DeviceKvLayout {
  self.kv_layout
}

///|
pub fn TwoNodePlan::model_identity(self : TwoNodePlan) -> @spec.ModelIdentity {
  self.model.identity()
}

///|
pub fn TwoNodePlan::topology(
  self : TwoNodePlan,
) -> @network_topology.TwoNodeTopology {
  self.topology
}

///|
pub fn TwoNodePlan::rank_plan(
  self : TwoNodePlan,
  rank : Int,
) -> @tensor_parallel_plan.TensorParallelRankPlan? {
  self.rank_plans.get(rank)
}

///|
pub fn TwoNodePlan::rank_budget(
  self : TwoNodePlan,
  rank : Int,
) -> RankMemoryBudget? {
  self.budgets.get(rank)
}

///|
pub fn TwoNodePlan::rank_capacity(
  self : TwoNodePlan,
  rank : Int,
) -> RankCapacity? {
  self.capacities.get(rank)
}

///|
pub fn TwoNodePlan::tokens_per_page(self : TwoNodePlan) -> Int {
  self.tokens_per_page
}

///|
pub fn TwoNodePlan::page_count(self : TwoNodePlan) -> Int {
  self.page_count
}
