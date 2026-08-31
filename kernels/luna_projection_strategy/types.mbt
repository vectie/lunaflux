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
  MatrixTile
} derive(Debug, Eq)

///|
pub(all) enum ProjectionStrategyRule {
  Shape
  Capability
  Bucket
  Measurement
  Candidate
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
  priv matrix_tile : ProjectionTile?
} derive(Debug, Eq)

///|
pub struct ProjectionStrategy {
  priv stable_id : Int
  priv phase : ProjectionPhase
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
/// One offline measurement. Runtime request handling never benchmarks.
pub struct ProjectionAutotuneRecord {
  priv device_class : String
  priv numeric_format : String
  priv shape : ProjectionShape
  priv strategy : ProjectionStrategy
  priv median_latency_ns : Int64
  priv sample_count : Int
} derive(Debug, Eq)
