///|
/// Backend-neutral arithmetic families used by tiled attention. Device
/// instruction names belong to the backend lowering package.
pub(all) enum AttentionTileCompute {
  AttentionScalarDot
  AttentionSubgroupDot
  AttentionMatrixDot
  AttentionMatrixQkPv
} derive(Debug, Eq)

///|
/// Backend-neutral memory pipelines for one query/key-value tile pair.
pub(all) enum AttentionTileMemory {
  AttentionDirectGlobal
  AttentionStagedShared
  AttentionAsyncStagedShared
} derive(Debug, Eq)

///|
pub(all) enum AttentionNumericType {
  AttentionBFloat16
  AttentionFloat16
  AttentionFloat32
  AttentionFloat8
} derive(Debug, Eq)

///|
pub(all) enum AttentionCandidateSource {
  AttentionOfflineAutotune
  AttentionStaticCostModel
} derive(Debug, Eq)

///|
pub struct AttentionMatrixShape {
  priv rows : Int
  priv columns : Int
  priv reduction : Int
} derive(Debug, Eq)

///|
/// Portable device limits consumed by the attention compiler. CUDA warp,
/// PTX, MFMA, XMX, and similar vendor concepts are intentionally absent.
pub struct AttentionCompilerCapabilities {
  priv subgroup_width : Int
  priv max_threads_per_block : Int
  priv max_shared_memory_bytes : Int64
  priv max_pipeline_stages : Int
  priv supports_async_copy : Bool
  priv matrix_shape : AttentionMatrixShape?
} derive(Debug, Eq)

///|
/// The bounded compilation problem for one serving bucket.
pub struct AttentionCompileProblem {
  priv shape : AttentionShape
  priv phase : AttentionExecutionPhase
  priv numeric_type : AttentionNumericType
  priv query_rows : Int
  priv context_tokens : Int
  priv causal : Bool
} derive(Debug, Eq)

///|
/// Tile geometry selected before device lowering.
pub struct AttentionTileShape {
  priv query_rows : Int
  priv key_value_rows : Int
  priv head_columns : Int
} derive(Debug, Eq)

///|
/// One deterministic compiler candidate. `stable_id` is part of offline
/// autotune identity and therefore must never be reused for different
/// semantics.
pub struct AttentionTileCandidate {
  priv stable_id : Int
  priv compute : AttentionTileCompute
  priv memory : AttentionTileMemory
  priv tile : AttentionTileShape
  priv subgroups_per_block : Int
  priv pipeline_stages : Int
  priv vector_bytes : Int
  priv shared_memory_bytes : Int64
} derive(Debug, Eq)

///|
pub struct AttentionBackendId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub struct AttentionTargetId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub struct AttentionAutotuneKey {
  priv backend : AttentionBackendId
  priv target : AttentionTargetId
  priv problem : AttentionCompileProblem
  priv candidate : AttentionTileCandidate
} derive(Debug, Eq)

///|
/// One offline measurement. Benchmarking and candidate scans never occur in
/// token-step execution.
pub struct AttentionAutotuneRecord {
  priv key : AttentionAutotuneKey
  priv median_latency_ns : Int64
  priv sample_count : Int
} derive(Debug, Eq)

///|
pub struct CompiledAttentionTilePlan {
  priv problem : AttentionCompileProblem
  priv candidate : AttentionTileCandidate
  priv source : AttentionCandidateSource
  priv static_cost : Int64
} derive(Debug, Eq)
