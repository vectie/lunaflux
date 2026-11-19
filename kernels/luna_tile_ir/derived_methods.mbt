///|
pub extend LunaTileCudaSourceDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileCudaSourceDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileCudaSourceDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileCudaTranslationDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileCudaTranslationDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileCudaTranslationDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileCudaTranslationUnit with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileCudaTranslationUnit with Eq::{not_equal, equal}

///|
pub extend LunaTileLimitRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileLimitRule with Eq::{not_equal, equal}

///|
pub extend LunaTileValidationRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileValidationRule with Eq::{not_equal, equal}

///|
pub extend LunaTileError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileError with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileTensorId with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorId with Hash::{hash, hash_combine}

///|
pub extend LunaTileStageId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileStageId with Eq::{not_equal, equal}

///|
pub extend LunaTileStageId with Hash::{hash, hash_combine}

///|
pub extend LunaTileConstraintId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileConstraintId with Eq::{not_equal, equal}

///|
pub extend LunaTileConstraintId with Hash::{hash, hash_combine}

///|
pub extend LunaTileEntryPointId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileEntryPointId with Eq::{not_equal, equal}

///|
pub extend LunaTileEntryPointId with Hash::{hash, hash_combine}

///|
pub extend LunaTileInstruction with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileInstruction with Eq::{not_equal, equal}

///|
pub extend LunaTileAotPlanningPolicy with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileAotPlanningPolicy with Eq::{not_equal, equal}

///|
pub extend LunaTileMemoryPlan with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileMemoryPlan with Eq::{not_equal, equal}

///|
pub extend LunaTileCudaAotDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileCudaAotDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileCudaAotDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileCudaAotInput with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileCudaAotInput with Eq::{not_equal, equal}

///|
pub extend LunaTileOperationSummary with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileOperationSummary with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorCorePolicy with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileTensorCorePolicy with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorCoreEligibility with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileTensorCoreEligibility with Eq::{not_equal, equal}

///|
pub extend LunaTileParallelExecutionPath with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileParallelExecutionPath with Eq::{not_equal, equal}

///|
pub extend LunaTileParallelPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileParallelPolicy with Eq::{not_equal, equal}

///|
pub extend LunaTileParallelMapping with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileParallelMapping with Eq::{not_equal, equal}

///|
pub extend LunaTileSharedPlacement with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileSharedPlacement with Eq::{not_equal, equal}

///|
pub extend LunaTileParallelSharedPlan with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileParallelSharedPlan with Eq::{not_equal, equal}

///|
pub extend LunaTileParallelDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileParallelDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileParallelDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileParallelCandidate with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileParallelCandidate with Eq::{not_equal, equal}

///|
pub extend LunaTileProgramInput with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileProgramInput with Eq::{not_equal, equal}

///|
pub extend LunaTileResidualAddSemantics with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaTileResidualAddSemantics with Eq::{not_equal, equal}

///|
pub extend LunaTileProgramDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileProgramDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileProgramDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileProgram with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileProgram with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorView with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileTensorView with Eq::{not_equal, equal}

///|
pub extend LunaTileAffineTerm with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileAffineTerm with Eq::{not_equal, equal}

///|
pub extend LunaTileAffineExpr with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileAffineExpr with Eq::{not_equal, equal}

///|
pub extend LunaTileRegion with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileRegion with Eq::{not_equal, equal}

///|
pub extend LunaTileDType with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileDType with Eq::{not_equal, equal}

///|
pub extend LunaTileAddressSpace with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileAddressSpace with Eq::{not_equal, equal}

///|
pub extend LunaTileBarrierScope with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileBarrierScope with Eq::{not_equal, equal}

///|
pub extend LunaTileReductionKind with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileReductionKind with Eq::{not_equal, equal}

///|
pub extend LunaTileElementwiseKind with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileElementwiseKind with Eq::{not_equal, equal}

///|
pub extend LunaTileConstraint with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileConstraint with Eq::{not_equal, equal}

///|
pub extend LunaTileLimits with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTileLimits with Eq::{not_equal, equal}
