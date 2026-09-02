///|
pub(all) enum ProjectionSemanticFamily {
  Qkv
  Dense
  GatedMlp
  LanguageModelHead
} derive(Debug, Eq)

///|
pub(all) enum ProjectionPhase {
  Decode
  Prefill
} derive(Debug, Eq)

///|
/// Backend-neutral execution classes. Device instruction names belong in lowering.
pub(all) enum ProjectionKernelClass {
  ScalarTile
  SubgroupGemv
  /// Matrix-unit GEMV obtained by padding the single live row inside a tile.
  /// The padding policy is semantic; the device backend chooses its matrix ISA.
  PaddedMatrixGemv
  MatrixTile
} derive(Debug, Eq)

///|
/// Backend-neutral workload class. Backends lower the selected class to their
/// own SIMD, subgroup, matrix, or library implementation.
pub(all) enum ProjectionStrategyClass {
  DecodeGemv
  SmallRowTiledGemm
  LargeRowTiledGemm
} derive(Debug, Eq)

///|
pub(all) enum ProjectionNumericType {
  ProjectionBFloat16
  ProjectionFloat16
  ProjectionFloat32
  ProjectionSignedInt8
  ProjectionFloat8
} derive(Debug, Eq)

///|
pub(all) enum ProjectionSelectionSource {
  OfflineAutotune
  StaticFallback
} derive(Debug, Eq)

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
pub(all) suberror ProjectionStrategyError {
  InvalidProjectionStrategy(ProjectionStrategyRule)
} derive(Debug, Eq)

///|
pub struct ProjectionShape {
  priv family : ProjectionSemanticFamily
  priv input_width : Int
  priv output_width : Int
  priv intermediate_width : Int
} derive(Debug, Eq)

///|
pub struct ProjectionTile {
  priv rows : Int
  priv columns : Int
  priv reduction : Int
} derive(Debug, Eq)

///|
pub struct ProjectionCapabilities {
  priv subgroup_width : Int
  priv padded_matrix_vector_tile : ProjectionTile?
  priv matrix_tile : ProjectionTile?
} derive(Debug, Eq)

///|
pub struct ProjectionBackendId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub struct ProjectionTargetId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub struct ProjectionStrategy {
  priv stable_id : Int
  priv phase : ProjectionPhase
  priv strategy_class : ProjectionStrategyClass
  priv kernel_class : ProjectionKernelClass
  priv row_bucket : Int
  priv tile : ProjectionTile
} derive(Debug, Eq)

///|
pub struct ProjectionDispatchPlan {
  priv shape : ProjectionShape
  priv decode : ProjectionStrategy
  priv prefill : ProjectionStrategy
} derive(Debug, Eq)

///|
pub struct ProjectionAutotuneKey {
  priv backend : ProjectionBackendId
  priv target : ProjectionTargetId
  priv numeric_type : ProjectionNumericType
  priv shape : ProjectionShape
  priv strategy : ProjectionStrategy
} derive(Debug, Eq)

///|
/// One offline measurement. Runtime request handling never benchmarks.
pub struct ProjectionAutotuneRecord {
  priv key : ProjectionAutotuneKey
  priv median_latency_ns : Int64
  priv sample_count : Int
} derive(Debug, Eq)

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
  priv maximum_rows : Int
  priv slots : FixedArray[ProjectionSelectedStrategy]
}
