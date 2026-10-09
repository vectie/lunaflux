///|
pub fn RankMemoryBudget::system_reserved_bytes(
  self : RankMemoryBudget,
) -> Int64 {
  self.system_reserved_bytes
}

///|
pub fn RankMemoryBudget::runtime_ceiling_bytes(
  self : RankMemoryBudget,
) -> Int64 {
  self.runtime_ceiling_bytes
}

///|
pub fn RankMemoryBudget::activation_workspace_bytes(
  self : RankMemoryBudget,
) -> Int64 {
  self.activation_workspace_bytes
}

///|
pub fn RankMemoryBudget::collective_bytes(self : RankMemoryBudget) -> Int64 {
  self.collective_bytes
}

///|
pub fn RankMemoryBudget::kv_bytes(self : RankMemoryBudget) -> Int64 {
  self.kv_bytes
}
