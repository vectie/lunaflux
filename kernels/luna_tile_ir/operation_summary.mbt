///|
/// Opaque generic operation inventory derived from an admitted program. It
/// exposes instruction semantics without asserting a model-family contract or
/// leaking mutable instruction storage.
pub struct LunaTileOperationSummary {
  priv copy_count : Int
  priv async_copy_count : Int
  priv barrier_count : Int
  priv pipeline_stage_count : Int
  priv mma_count : Int
  priv reduction_count : Int
  priv elementwise_count : Int
} derive(Debug, Eq)

///|
pub fn LunaTileOperationSummary::copy_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.copy_count
}

///|
pub fn LunaTileOperationSummary::async_copy_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.async_copy_count
}

///|
pub fn LunaTileOperationSummary::barrier_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.barrier_count
}

///|
pub fn LunaTileOperationSummary::pipeline_stage_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.pipeline_stage_count
}

///|
pub fn LunaTileOperationSummary::mma_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.mma_count
}

///|
pub fn LunaTileOperationSummary::reduction_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.reduction_count
}

///|
pub fn LunaTileOperationSummary::elementwise_count(
  self : LunaTileOperationSummary,
) -> Int {
  self.elementwise_count
}

///|
pub fn LunaTileProgram::operation_summary(
  self : LunaTileProgram,
) -> LunaTileOperationSummary {
  let mut copy_count = 0
  let mut async_copy_count = 0
  let mut barrier_count = 0
  let mut pipeline_stage_count = 0
  let mut mma_count = 0
  let mut reduction_count = 0
  let mut elementwise_count = 0
  for instruction in self.instructions {
    match instruction {
      LunaTileCopy(_, _) => copy_count += 1
      LunaTileAsyncCopy(_, _, _) => async_copy_count += 1
      LunaTileBarrier(_, _) => barrier_count += 1
      LunaTilePipelineStage(_) => pipeline_stage_count += 1
      LunaTileMma(..) => mma_count += 1
      LunaTileReduce(_, ..) => reduction_count += 1
      LunaTileElementwise(_, ..) => elementwise_count += 1
    }
  }
  {
    copy_count,
    async_copy_count,
    barrier_count,
    pipeline_stage_count,
    mma_count,
    reduction_count,
    elementwise_count,
  }
}
