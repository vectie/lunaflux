///|
/// Observable projection rows. `QueryRowEnds` is legal only for a final
/// language-model head whose consumer requests one next-token distribution
/// per packed query row.
pub(all) enum ProjectionResultDemand {
  AllTokenRows
  QueryRowEnds
} derive(Debug, Eq)

///|
pub(all) enum ProjectionTileValueKind {
  ProjectionInputRows
  ProjectionSelectedRows
  ProjectionWeights
  ProjectionIntermediateRows
  ProjectionAccumulator
  ProjectionNormalizedRows
  ProjectionPositionedRows
  ProjectionOutputRows
} derive(Debug, Eq, Hash)

///|
pub(all) enum ProjectionTileWeightRole {
  ProjectionDenseWeight
  ProjectionGateWeight
  ProjectionUpWeight
  ProjectionDownWeight
  ProjectionQueryNormWeight
  ProjectionKeyNormWeight
} derive(Debug, Eq, Hash)

///|
/// Backend-neutral parameters for an attention-ingress projection epilogue.
/// The compiler sees heads and scalar semantics; subgroup and device details
/// are introduced only by a terminal lowering.
pub struct ProjectionAttentionIngressEpilogue {
  priv query_heads : Int32
  priv kv_heads : Int32
  priv head_dimension : Int32
  priv norm_epsilon : Double
  priv rope_theta : Double
} derive(Debug, Eq)

///|
pub(all) enum ProjectionTileEpilogue {
  ProjectionStoreRows
  ProjectionAttentionIngress(ProjectionAttentionIngressEpilogue)
} derive(Debug, Eq)

///|
pub struct ProjectionTileValueId(Int32) derive(Debug, Eq, Hash)

///|
pub(all) enum ProjectionTileExpression {
  ProjectionTileInput
  ProjectionTileWeight(ProjectionTileWeightRole)
  ProjectionSelectQueryRowEnds(ProjectionTileValueId)
  ProjectionDot(ProjectionTileValueId, ProjectionTileValueId)
  ProjectionSiluGate(ProjectionTileValueId, ProjectionTileValueId)
  ProjectionQkHeadRmsNorm(
    ProjectionTileValueId,
    ProjectionTileValueId,
    ProjectionTileValueId
  )
  ProjectionPositionedRotary(ProjectionTileValueId)
  ProjectionStoreAndCommitPagedKv(ProjectionTileValueId)
  ProjectionStore(ProjectionTileValueId)
} derive(Debug, Eq, Hash)

///|
pub struct ProjectionTileBinding {
  priv id : ProjectionTileValueId
  priv kind : ProjectionTileValueKind
  priv expression : ProjectionTileExpression
} derive(Debug, Eq)

///|
pub struct ProjectionTileProgramDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ProjectionTileProgram {
  priv concatenated_weights : ReadOnlyArray[ProjectionRowSegment]
  priv shape : @luna_projection_strategy.ProjectionShape
  priv demand : ProjectionResultDemand
  priv epilogue : ProjectionTileEpilogue
  priv maximum_rows : Int32
  priv maximum_tokens : Int32
  priv bindings : ReadOnlyArray[ProjectionTileBinding]
  priv root : ProjectionTileValueId
  priv canonical : Bytes
  priv digest : ProjectionTileProgramDigest
} derive(Debug, Eq)

///|
pub(all) enum ProjectionTileRewriteRule {
  PruneUnobservedTokenRows
  HoistSelectedInputTile
  ReuseInputTileAcrossOutputTiles
  FuseSiblingInputTraversals
  ReuseIntermediateTileAcrossOutputTiles
  FuseAttentionIngressEpilogue
  ElideProjectedQkvRoundTrip
  HoistRotaryInverseFrequencies
  FuseRotaryPairEvaluation
} derive(Debug, Eq)

///|
pub struct ProjectionTileRewriteStep {
  priv rule : ProjectionTileRewriteRule
  priv changed : Bool
} derive(Debug, Eq)

///|
pub struct ProjectionTileOptimizationDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ProjectionTileOptimization {
  priv program_digest : ProjectionTileProgramDigest
  priv trace : ReadOnlyArray[ProjectionTileRewriteStep]
  priv canonical : Bytes
  priv digest : ProjectionTileOptimizationDigest
} derive(Debug, Eq)

///|
pub(all) enum ProjectionIterationKind {
  ProjectionParallelMap
  ProjectionSequentialFold
} derive(Debug, Eq)

///|
pub(all) enum ProjectionIterationAxis {
  ProjectionRowAxis
  ProjectionIntermediateColumnAxis
  ProjectionOutputColumnAxis
  ProjectionReductionAxis
  ProjectionIntermediateReductionAxis
  ProjectionHeadAxis
  ProjectionHeadComponentAxis
} derive(Debug, Eq)

///|
/// Backend-neutral handling for the final, partially populated projection
/// row tile. Matrix backends lower zero padding with their own local-memory
/// and matrix-instruction vocabulary; scalar backends retain the live extent.
pub(all) enum ProjectionRowTailPolicy {
  ScalarRowTail
  ZeroPaddedMatrixRowTail
} derive(Debug, Eq)

///|
pub struct ProjectionLoopCombinator {
  priv kind : ProjectionIterationKind
  priv axis : ProjectionIterationAxis
  priv extent : Int32
  priv tile : Int32
} derive(Debug, Eq)

///|
pub struct ProjectionTileScheduleDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ProjectionTileSchedule {
  priv program_digest : ProjectionTileProgramDigest
  priv optimization_digest : ProjectionTileOptimizationDigest
  priv demand : ProjectionResultDemand
  priv loops : ReadOnlyArray[ProjectionLoopCombinator]
  priv output_tiles_per_workgroup : Int32
  priv selected_input_row_resident : Bool
  priv row_tail_policy : ProjectionRowTailPolicy
  priv reuse_intermediate_tile : Bool
  priv intermediate_pipeline : ProjectionFoldPipeline?
  priv sibling_pipeline : ProjectionFoldPipeline?
  priv matrix_pipeline : ProjectionFoldPipeline?
  priv fuse_attention_ingress_epilogue : Bool
  priv hoist_rotary_inverse_frequencies : Bool
  priv fuse_rotary_pairs : Bool
  priv single_row_reduction : ProjectionSingleRowReduction
  priv canonical : Bytes
  priv digest : ProjectionTileScheduleDigest
} derive(Debug, Eq)

///|
pub struct ProjectionTileCompilerRequest {
  priv ingress_row_tiles : Int32
  priv ingress_head_tiles : Int32
  priv dispatch : @luna_projection_strategy.ProjectionDispatchPlan
  priv maximum_rows : Int32
  priv maximum_tokens : Int32
  priv demand : ProjectionResultDemand
  priv epilogue : ProjectionTileEpilogue
  priv single_row_numerics : ProjectionSingleRowNumerics
  priv fold_choices : ReadOnlyArray[ProjectionFoldChoice]
  priv concatenated_weights : ReadOnlyArray[ProjectionRowSegment]
} derive(Debug, Eq)

///|
pub struct SelectedProjectionTileCompilation {
  priv request : ProjectionTileCompilerRequest
} derive(Debug, Eq)

///|
pub struct SemanticProjectionTileCompilation {
  priv selected : SelectedProjectionTileCompilation
  priv program : ProjectionTileProgram
} derive(Debug, Eq)

///|
pub struct OptimizedProjectionTileCompilation {
  priv semantic : SemanticProjectionTileCompilation
  priv normalized : ProjectionTileNormalization
  priv optimization : ProjectionTileOptimization
} derive(Debug, Eq)

///|
pub struct ProjectionTileCompilationDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ScheduledProjectionTileCompilation {
  priv optimized : OptimizedProjectionTileCompilation
  priv schedule : ProjectionTileSchedule
  priv canonical : Bytes
  priv digest : ProjectionTileCompilationDigest
} derive(Debug, Eq)

///|
pub(all) enum ProjectionTileCompilerRule {
  InvalidProblem
  InvalidDemand
  InvalidSemanticProgram
  InvalidRewritePipeline
  InvalidSchedule
} derive(Debug, Eq)

///|
pub(all) suberror ProjectionTileCompilerError {
  InvalidProjectionTileCompilation(ProjectionTileCompilerRule)
} derive(Debug, Eq)

///|
pub extend ProjectionResultDemand with @debug.Debug::{to_repr}

///|
pub extend ProjectionResultDemand with Eq::{not_equal, equal}

///|
pub extend ProjectionTileValueKind with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileValueKind with Eq::{not_equal, equal}

///|
pub extend ProjectionTileValueKind with Hash::{hash, hash_combine}

///|
pub extend ProjectionTileWeightRole with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileWeightRole with Eq::{not_equal, equal}

///|
pub extend ProjectionTileWeightRole with Hash::{hash, hash_combine}

///|
pub extend ProjectionAttentionIngressEpilogue with @debug.Debug::{to_repr}

///|
pub extend ProjectionAttentionIngressEpilogue with Eq::{not_equal, equal}

///|
pub extend ProjectionTileEpilogue with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileEpilogue with Eq::{not_equal, equal}

///|
pub extend ProjectionTileValueId with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileValueId with Eq::{not_equal, equal}

///|
pub extend ProjectionTileValueId with Hash::{hash, hash_combine}

///|
pub extend ProjectionTileExpression with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileExpression with Eq::{not_equal, equal}

///|
pub extend ProjectionTileExpression with Hash::{hash, hash_combine}

///|
pub extend ProjectionTileBinding with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileBinding with Eq::{not_equal, equal}

///|
pub extend ProjectionTileProgramDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileProgramDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionTileProgramDigest with Hash::{hash, hash_combine}

///|
pub extend ProjectionTileProgram with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileProgram with Eq::{not_equal, equal}

///|
pub extend ProjectionTileRewriteRule with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileRewriteRule with Eq::{not_equal, equal}

///|
pub extend ProjectionTileRewriteStep with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileRewriteStep with Eq::{not_equal, equal}

///|
pub extend ProjectionTileOptimizationDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileOptimizationDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionTileOptimizationDigest with Hash::{hash, hash_combine}

///|
pub extend ProjectionTileOptimization with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileOptimization with Eq::{not_equal, equal}

///|
pub extend ProjectionIterationKind with @debug.Debug::{to_repr}

///|
pub extend ProjectionIterationKind with Eq::{not_equal, equal}

///|
pub extend ProjectionIterationAxis with @debug.Debug::{to_repr}

///|
pub extend ProjectionIterationAxis with Eq::{not_equal, equal}

///|
pub extend ProjectionRowTailPolicy with @debug.Debug::{to_repr}

///|
pub extend ProjectionRowTailPolicy with Eq::{not_equal, equal}

///|
pub extend ProjectionLoopCombinator with @debug.Debug::{to_repr}

///|
pub extend ProjectionLoopCombinator with Eq::{not_equal, equal}

///|
pub extend ProjectionTileScheduleDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileScheduleDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionTileScheduleDigest with Hash::{hash, hash_combine}

///|
pub extend ProjectionTileSchedule with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileSchedule with Eq::{not_equal, equal}

///|
pub extend ProjectionTileCompilerRequest with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileCompilerRequest with Eq::{not_equal, equal}

///|
pub extend SelectedProjectionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend SelectedProjectionTileCompilation with Eq::{not_equal, equal}

///|
pub extend SemanticProjectionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend SemanticProjectionTileCompilation with Eq::{not_equal, equal}

///|
pub extend OptimizedProjectionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend OptimizedProjectionTileCompilation with Eq::{not_equal, equal}

///|
pub extend ProjectionTileCompilationDigest with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileCompilationDigest with Eq::{not_equal, equal}

///|
pub extend ProjectionTileCompilationDigest with Hash::{hash, hash_combine}

///|
pub extend ScheduledProjectionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend ScheduledProjectionTileCompilation with Eq::{not_equal, equal}

///|
pub extend ProjectionTileCompilerRule with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileCompilerRule with Eq::{not_equal, equal}

///|
pub extend ProjectionTileCompilerError with @debug.Debug::{to_repr}

///|
pub extend ProjectionTileCompilerError with Eq::{not_equal, equal}
