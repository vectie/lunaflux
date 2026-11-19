///|
pub(all) enum ProjectionSemanticFamily {
  Qkv
  Dense
  GatedMlp
  LanguageModelHead
} derive(Debug, Eq)

///|
pub extend ProjectionSemanticFamily with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ProjectionSemanticFamily with Eq::{not_equal, equal}

///|
pub(all) enum ProjectionPhase {
  Decode
  Prefill
} derive(Debug, Eq)

///|
pub extend ProjectionPhase with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionPhase with Eq::{not_equal, equal}

///|
/// Backend-neutral execution classes. Device instruction names belong in lowering.
pub(all) enum ProjectionKernelClass {
  ScalarTile
  SubgroupGemv
  MatrixTile
} derive(Debug, Eq)

///|
pub extend ProjectionKernelClass with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionKernelClass with Eq::{not_equal, equal}

///|
/// Backend-neutral workload class. Backends lower the selected class to their
/// own SIMD, subgroup, matrix, or library implementation.
pub(all) enum ProjectionStrategyClass {
  DecodeGemv
  SmallRowTiledGemm
  LargeRowTiledGemm
} derive(Debug, Eq)

///|
pub extend ProjectionStrategyClass with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionStrategyClass with Eq::{not_equal, equal}

///|
pub(all) enum ProjectionNumericType {
  ProjectionBFloat16
  ProjectionFloat16
  ProjectionFloat32
  ProjectionSignedInt8
  ProjectionFloat8
} derive(Debug, Eq)

///|
pub extend ProjectionNumericType with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionNumericType with Eq::{not_equal, equal}

///|
pub(all) enum ProjectionSelectionSource {
  OfflineAutotune
  StaticFallback
} derive(Debug, Eq)

///|
pub extend ProjectionSelectionSource with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ProjectionSelectionSource with Eq::{not_equal, equal}

///|
pub(all) enum ProjectionStrategyRule {
  Shape
  Capability
  Bucket
  Measurement
  Candidate
  Backend
  Target
  Duplicate
  Conflict
} derive(Debug, Eq)

///|
pub extend ProjectionStrategyRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionStrategyRule with Eq::{not_equal, equal}

///|
pub(all) suberror ProjectionStrategyError {
  InvalidProjectionStrategy(ProjectionStrategyRule)
} derive(Debug, Eq)

///|
pub extend ProjectionStrategyError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionStrategyError with Eq::{not_equal, equal}

///|
pub struct ProjectionShape {
  priv family : ProjectionSemanticFamily
  priv input_width : Int32
  priv output_width : Int32
  priv intermediate_width : Int32
} derive(Debug, Eq)

///|
pub extend ProjectionShape with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionShape with Eq::{not_equal, equal}

///|
pub struct ProjectionTile {
  priv rows : Int32
  priv columns : Int32
  priv reduction : Int32
} derive(Debug, Eq)

///|
pub extend ProjectionTile with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionTile with Eq::{not_equal, equal}

///|
pub struct ProjectionCapabilities {
  priv subgroup_width : Int32
  priv matrix_tile : ProjectionTile?
} derive(Debug, Eq)

///|
pub extend ProjectionCapabilities with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionCapabilities with Eq::{not_equal, equal}

///|
pub struct ProjectionBackendId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub extend ProjectionBackendId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionBackendId with Eq::{not_equal, equal}

///|
pub struct ProjectionTargetId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub extend ProjectionTargetId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionTargetId with Eq::{not_equal, equal}

///|
pub struct ProjectionStrategy {
  priv stable_id : Int32
  priv phase : ProjectionPhase
  priv strategy_class : ProjectionStrategyClass
  priv kernel_class : ProjectionKernelClass
  priv row_bucket : Int32
  priv tile : ProjectionTile
  priv output_tiles_per_workgroup : Int32
  priv input_residency : ProjectionInputResidency
} derive(Debug, Eq)

///|
pub extend ProjectionStrategy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionStrategy with Eq::{not_equal, equal}

///|
pub struct ProjectionDispatchPlan {
  priv shape : ProjectionShape
  priv decode : ProjectionStrategy
  priv prefill : ProjectionStrategy
} derive(Debug, Eq)

///|
pub extend ProjectionDispatchPlan with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionDispatchPlan with Eq::{not_equal, equal}

///|
pub struct ProjectionAutotuneKey {
  priv backend : ProjectionBackendId
  priv target : ProjectionTargetId
  priv numeric_type : ProjectionNumericType
  priv shape : ProjectionShape
  priv strategy : ProjectionStrategy
} derive(Debug, Eq)

///|
pub extend ProjectionAutotuneKey with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ProjectionAutotuneKey with Eq::{not_equal, equal}

///|
/// One offline measurement. Runtime request handling never benchmarks.
pub struct ProjectionAutotuneRecord {
  priv key : ProjectionAutotuneKey
  priv median_latency_ns : Int64
  priv sample_count : Int32
} derive(Debug, Eq)

///|
pub extend ProjectionAutotuneRecord with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ProjectionAutotuneRecord with Eq::{not_equal, equal}

///|
#valtype
pub struct ProjectionSelectedStrategy {
  priv strategy : ProjectionStrategy
  priv source : ProjectionSelectionSource
}

///|
/// Startup-built constant-time dispatch table. Request-time selection performs
/// only bounds checks, a fixed comparison ladder, and one FixedArray lookup.
pub struct ProjectionRuntimeDispatch {
  priv backend : ProjectionBackendId
  priv target : ProjectionTargetId
  priv numeric_type : ProjectionNumericType
  priv shape : ProjectionShape
  priv maximum_rows : Int32
  priv slots : FixedArray[ProjectionSelectedStrategy]
}
