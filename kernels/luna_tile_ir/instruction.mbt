///|
pub(all) enum LunaTileInstruction {
  LunaTileCopy(LunaTileRegion, LunaTileRegion)
  LunaTileAsyncCopy(LunaTileRegion, LunaTileRegion, LunaTileStageId)
  LunaTileBarrier(LunaTileStageId, LunaTileBarrierScope)
  LunaTilePipelineStage(LunaTileStageId)
  LunaTileMma(
    destination~ : LunaTileTensorId,
    left~ : LunaTileTensorId,
    right~ : LunaTileTensorId,
    accumulator~ : LunaTileTensorId
  )
  LunaTileReduce(
    LunaTileReductionKind,
    destination~ : LunaTileTensorId,
    source~ : LunaTileTensorId,
    axis~ : Int
  )
  LunaTileElementwise(
    LunaTileElementwiseKind,
    destination~ : LunaTileTensorId,
    inputs~ : ReadOnlyArray[LunaTileTensorId]
  )
} derive(Debug, Eq)

///|
pub fn LunaTileInstruction::elementwise(
  kind : LunaTileElementwiseKind,
  destination : LunaTileTensorId,
  inputs : ArrayView[LunaTileTensorId],
) -> LunaTileInstruction {
  LunaTileElementwise(
    kind,
    destination~,
    inputs=ReadOnlyArray::from_array(inputs),
  )
}
