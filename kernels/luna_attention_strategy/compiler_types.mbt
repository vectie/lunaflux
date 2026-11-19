///|
/// Backend-neutral arithmetic families used by tiled attention. Device
/// instruction names belong to the backend lowering package.
pub(all) enum AttentionTileCompute {
  AttentionScalarDot
  AttentionSubgroupDot
  /// One subgroup owns each query head in a grouped-query attention group.
  /// The block shares the corresponding K/V tile across those subgroups.
  AttentionGroupedQuerySubgroupDot
  /// Multiple subgroups cooperate on each query head while the complete GQA
  /// group shares one K/V tile.
  AttentionGroupedQuerySplitSubgroupDot
  /// Tile-local F32 max/probabilities followed by one state merge per tile.
  /// This is deliberately not the ordered per-key recurrence.
  AttentionGroupedQueryBlockwiseDot
  /// Two independent score owners share each subgroup, then join one tile
  /// state. The narrower QK reduction has its own declared numerical law.
  AttentionGroupedQueryDualScoreBlockwiseDot
  /// Explicit F32 contraction in the blockwise dot/value/state folds; not
  /// permission for global reassociation or approximate exponentiation.
  AttentionGroupedQueryFmaBlockwiseDot
  /// Explicitly partition score ownership without changing the F32 PV law.
  /// Owner count and multiply/add contraction both participate in identity.
  AttentionGroupedQueryOwnedBlockwiseDot(
    score_owners~ : Int32,
    contraction~ : Bool
  )
  /// Matrix rows represent query heads, not additional sequence tokens.
  AttentionGroupedQueryMatrixQkPv
  AttentionMatrixDot
  AttentionMatrixQkPv
  /// Explicit opt-in approximate base-two exponential; strict law retained.
  AttentionMatrixQkPvApproxExp
} derive(Debug, Eq)

///|
pub extend AttentionTileCompute with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileCompute with Eq::{not_equal, equal}

///|
/// Backend-neutral memory pipelines for one query/key-value tile pair.
pub(all) enum AttentionTileMemory {
  AttentionDirectGlobal
  AttentionStagedShared
  AttentionAsyncStagedShared
} derive(Debug, Eq)

///|
pub extend AttentionTileMemory with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileMemory with Eq::{not_equal, equal}

///|
pub(all) enum AttentionNumericType {
  AttentionBFloat16
  AttentionFloat16
  AttentionFloat32
  AttentionFloat8
} derive(Debug, Eq)

///|
pub extend AttentionNumericType with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionNumericType with Eq::{not_equal, equal}

///|
pub(all) enum AttentionCandidateSource {
  AttentionOfflineAutotune
  AttentionStaticCostModel
} derive(Debug, Eq)

///|
pub extend AttentionCandidateSource with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionCandidateSource with Eq::{not_equal, equal}

///|
pub struct AttentionMatrixShape {
  priv rows : Int32
  priv columns : Int32
  priv reduction : Int32
} derive(Debug, Eq)

///|
pub extend AttentionMatrixShape with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionMatrixShape with Eq::{not_equal, equal}

///|
/// Portable device limits consumed by the attention compiler. CUDA warp,
/// PTX, MFMA, XMX, and similar vendor concepts are intentionally absent.
pub struct AttentionCompilerCapabilities {
  priv subgroup_width : Int32
  priv max_threads_per_block : Int32
  priv max_shared_memory_bytes : Int64
  priv max_pipeline_stages : Int32
  priv supports_async_copy : Bool
  /// Portable occupancy objective used by schedule rewrites. This is a count
  /// of independent workgroups, not a vendor compute-unit or warp concept.
  priv parallel_workgroup_target : Int32
  priv matrix_shape : AttentionMatrixShape?
} derive(Debug, Eq)

///|
pub extend AttentionCompilerCapabilities with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionCompilerCapabilities with Eq::{not_equal, equal}

///|
/// The bounded compilation problem for one serving bucket.
pub struct AttentionCompileProblem {
  priv shape : AttentionShape
  priv phase : AttentionExecutionPhase
  priv numeric_type : AttentionNumericType
  /// Number of independent sequences in the launch bucket. This is distinct
  /// from `query_rows`: decode has one query token per sequence, while the
  /// bucket can still contain many sequences.
  priv batch_rows : Int32
  priv query_rows : Int32
  priv context_tokens : Int32
  priv causal : Bool
  priv allow_alternative_softmax_reduction : Bool
  priv allow_approximate_exponential : Bool
  priv key_value_read_view : AttentionKeyValueReadView
  priv required_fragment_ownership : AttentionFragmentOwnership?
} derive(Debug, Eq)

///|
pub extend AttentionCompileProblem with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionCompileProblem with Eq::{not_equal, equal}

///|
/// Tile geometry selected before device lowering.
pub struct AttentionTileShape {
  priv query_rows : Int32
  priv key_value_rows : Int32
  priv head_columns : Int32
} derive(Debug, Eq)

///|
pub extend AttentionTileShape with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileShape with Eq::{not_equal, equal}

///|
/// One deterministic compiler candidate. `stable_id` is part of offline
/// autotune identity and therefore must never be reused for different
/// semantics.
pub struct AttentionTileCandidate {
  priv stable_id : Int32
  priv compute : AttentionTileCompute
  priv memory : AttentionTileMemory
  priv tile : AttentionTileShape
  priv subgroups_per_block : Int32
  priv pipeline_stages : Int32
  priv vector_bytes : Int32
  priv shared_memory_bytes : Int64
  /// Immutable query reduction prefix retained across the KV fold. Zero
  /// retains the per-fold load program. This is a semantic dimension extent,
  /// not a backend register count or a model-specific tuning constant.
  priv retained_query_dimensions : Int32
  priv joint_query_consumers : Bool
  /// Zero preserves the original provider ownership. Positive values bind
  /// consumer-contiguous read ownership and an ordered read-ahead frontier.
  priv right_operand_read_slots : Int32
} derive(Debug, Eq)

///|
pub extend AttentionTileCandidate with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileCandidate with Eq::{not_equal, equal}

///|
pub struct AttentionBackendId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub extend AttentionBackendId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionBackendId with Eq::{not_equal, equal}

///|
pub struct AttentionTargetId {
  priv stable_name : String
} derive(Debug, Eq)

///|
pub extend AttentionTargetId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTargetId with Eq::{not_equal, equal}

///|
pub struct AttentionAutotuneKey {
  priv backend : AttentionBackendId
  priv target : AttentionTargetId
  priv problem : AttentionCompileProblem
  priv candidate : AttentionTileCandidate
} derive(Debug, Eq)

///|
pub extend AttentionAutotuneKey with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionAutotuneKey with Eq::{not_equal, equal}

///|
/// One offline measurement. Benchmarking and candidate scans never occur in
/// token-step execution.
pub struct AttentionAutotuneRecord {
  priv key : AttentionAutotuneKey
  priv median_latency_ns : Int64
  priv sample_count : Int32
} derive(Debug, Eq)

///|
pub extend AttentionAutotuneRecord with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionAutotuneRecord with Eq::{not_equal, equal}

///|
pub struct CompiledAttentionTilePlan {
  priv problem : AttentionCompileProblem
  priv candidate : AttentionTileCandidate
  priv source : AttentionCandidateSource
  priv static_cost : Int64
} derive(Debug, Eq)

///|
pub extend CompiledAttentionTilePlan with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend CompiledAttentionTilePlan with Eq::{not_equal, equal}
