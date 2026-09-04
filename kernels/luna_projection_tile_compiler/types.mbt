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
  ProjectionAccumulator
  ProjectionOutputRows
} derive(Debug, Eq)

///|
pub struct ProjectionTileValueId(Int) derive(Debug, Eq, Hash)

///|
pub(all) enum ProjectionTileExpression {
  ProjectionTileInput
  ProjectionTileWeight
  ProjectionSelectQueryRowEnds(ProjectionTileValueId)
  ProjectionDot(ProjectionTileValueId, ProjectionTileValueId)
  ProjectionStore(ProjectionTileValueId)
} derive(Debug, Eq)

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
  priv shape : @luna_projection_strategy.ProjectionShape
  priv demand : ProjectionResultDemand
  priv maximum_rows : Int
  priv maximum_tokens : Int
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
  ProjectionOutputColumnAxis
  ProjectionReductionAxis
} derive(Debug, Eq)

///|
pub struct ProjectionLoopCombinator {
  priv kind : ProjectionIterationKind
  priv axis : ProjectionIterationAxis
  priv extent : Int
  priv tile : Int
} derive(Debug, Eq)

///|
pub struct ProjectionTileScheduleDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ProjectionTileSchedule {
  priv program_digest : ProjectionTileProgramDigest
  priv optimization_digest : ProjectionTileOptimizationDigest
  priv demand : ProjectionResultDemand
  priv loops : ReadOnlyArray[ProjectionLoopCombinator]
  priv reuse_input_tile : Bool
  priv canonical : Bytes
  priv digest : ProjectionTileScheduleDigest
} derive(Debug, Eq)

///|
pub struct ProjectionTileCompilerRequest {
  priv dispatch : @luna_projection_strategy.ProjectionDispatchPlan
  priv maximum_rows : Int
  priv maximum_tokens : Int
  priv demand : ProjectionResultDemand
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
