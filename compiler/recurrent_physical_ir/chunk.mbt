///|
/// Frame-local triangular factors are immutable after prepare. The incoming
/// request state remains read-only until all transformed values are published.
pub struct ChunkDeltaPlan {
  priv numerical : @precision.RecurrentDeltaPrecision
  priv value_tile : Int
  priv workspace : Int64
  priv per_head : Int64
  priv local_factors : Bool
  priv local_bytes : Int
}

///|
pub fn ChunkDeltaPlan::new(
  numerical : @precision.RecurrentDeltaPrecision,
  value_tile~ : Int,
  maximum_workspace_bytes~ : Int64,
  maximum_local_storage_bytes? : Int = 0,
) -> Self raise @precision.PrecisionPlanError {
  let (heads, keys, values) = numerical.geometry()
  let (rows, sequences, _) = numerical.capacity()
  guard numerical.evaluation() is ChunkTriangularFloat32 &&
    value_tile > 0 &&
    value_tile <= values &&
    rows <= 64 &&
    keys <= 256 &&
    values <= 256 &&
    maximum_workspace_bytes > 0L &&
    maximum_local_storage_bytes >= 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  // Six normalized/decayed operand planes; causal L, QK, inverse;
  // transformed values and incoming-state output projection.
  let r = rows.to_int64()
  let per_head = 6L * r * keys.to_int64() +
    3L * r * r +
    2L * r * values.to_int64()
  guard heads.to_int64() <= 9223372036854775807L / sequences.to_int64() &&
    heads.to_int64() * sequences.to_int64() <=
    maximum_workspace_bytes / (per_head * 4L) else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let local_bytes = 8 * keys * (rows + 1) + 8 * rows * (rows + 1) + 8 * rows
  {
    numerical,
    value_tile,
    per_head,
    local_factors: local_bytes <= maximum_local_storage_bytes,
    local_bytes,
    workspace: per_head * 4L * heads.to_int64() * sequences.to_int64(),
  }
}

///|
pub fn ChunkDeltaPlan::numerical(
  self : Self,
) -> @precision.RecurrentDeltaPrecision {
  self.numerical
}

///|
pub fn ChunkDeltaPlan::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}

///|
pub fn ChunkDeltaPlan::head_floats(self : Self) -> Int64 {
  self.per_head
}

///|
pub fn ChunkDeltaPlan::value_tile(self : Self) -> Int {
  self.value_tile
}

///|
/// Transposed key/gate tiles and padded causal matrices fit together or stay
/// in frame storage. This storage choice cannot change the numerical fold.
pub fn ChunkDeltaPlan::local_factors(self : Self) -> Bool {
  self.local_factors
}

///|
pub fn ChunkDeltaPlan::local_storage_bytes(self : Self) -> Int {
  if self.local_factors {
    self.local_bytes
  } else {
    0
  }
}

///|
pub fn ChunkDeltaPlan::stage_count(self : Self) -> Int {
  ignore(self)
  3
}

///|
pub fn ChunkDeltaPlan::state_effect(
  self : Self,
  stage : Int,
) -> @precision.NumericEffect {
  ignore(self)
  if stage == 2 {
    CacheCommit
  } else {
    StepWrite
  }
}
