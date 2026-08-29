///|
fn luna_tile_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub(all) enum LunaTileDType {
  BFloat16
  Float16
  Float32
  Int8
  Int32
} derive(Debug, Eq)

///|
pub(all) enum LunaTileAddressSpace {
  Global
  Shared
  Register
} derive(Debug, Eq)

///|
pub(all) enum LunaTileBarrierScope {
  Warp
  Block
} derive(Debug, Eq)

///|
pub(all) enum LunaTileReductionKind {
  Sum
  Maximum
} derive(Debug, Eq)

///|
pub(all) enum LunaTileElementwiseKind {
  Add
  Multiply
  Maximum
  Silu
} derive(Debug, Eq)

///|
pub(all) enum LunaTileConstraint {
  LunaTileDimensionEquals(LunaTileConstraintId, LunaTileTensorId, Int, Int64)
  LunaTileDimensionMultipleOf(
    LunaTileConstraintId,
    LunaTileTensorId,
    Int,
    Int64
  )
  LunaTileAlignmentAtLeast(LunaTileConstraintId, LunaTileTensorId, Int64)
} derive(Debug, Eq)

///|
pub struct LunaTileLimits {
  priv max_tensors : Int
  priv max_instructions : Int
  priv max_rank : Int
  priv max_loop_rank : Int
  priv max_constraints : Int
  priv max_canonical_bytes : Int
  priv max_shared_memory_bytes : Int64
} derive(Debug, Eq)

///|
pub fn LunaTileLimits::new(
  max_tensors~ : Int,
  max_instructions~ : Int,
  max_rank~ : Int,
  max_loop_rank~ : Int,
  max_constraints~ : Int,
  max_canonical_bytes~ : Int,
  max_shared_memory_bytes~ : Int64,
) -> LunaTileLimits raise LunaTileError {
  if max_tensors <= 0 {
    raise InvalidLimit(TensorCount)
  }
  if max_instructions <= 0 {
    raise InvalidLimit(InstructionCount)
  }
  if max_rank <= 0 {
    raise InvalidLimit(Rank)
  }
  if max_loop_rank <= 0 {
    raise InvalidLimit(LoopRank)
  }
  if max_constraints < 0 {
    raise InvalidLimit(ConstraintCount)
  }
  if max_canonical_bytes <= 0 {
    raise InvalidLimit(CanonicalBytes)
  }
  if max_shared_memory_bytes < 0L {
    raise InvalidLimit(SharedMemoryBytes)
  }
  {
    max_tensors,
    max_instructions,
    max_rank,
    max_loop_rank,
    max_constraints,
    max_canonical_bytes,
    max_shared_memory_bytes,
  }
}

///|
pub fn LunaTileLimits::max_shared_memory_bytes(self : LunaTileLimits) -> Int64 {
  self.max_shared_memory_bytes
}

///|
fn luna_tile_dtype_bytes(dtype : LunaTileDType) -> Int64 {
  match dtype {
    BFloat16 | Float16 => 2L
    Float32 | Int32 => 4L
    Int8 => 1L
  }
}
