///|
/// Every influence on strategy selection is an immutable request value.
/// Filesystem access, benchmarking, and device probing happen before this
/// functional compiler core is called.
pub struct AttentionTileCompilerRequest {
  priv backend : @luna_attention_strategy.AttentionBackendId
  priv target : @luna_attention_strategy.AttentionTargetId
  priv problem : @luna_attention_strategy.AttentionCompileProblem
  priv capabilities : @luna_attention_strategy.AttentionCompilerCapabilities
  priv autotune_records : ReadOnlyArray[
    @luna_attention_strategy.AttentionAutotuneRecord,
  ]
} derive(Debug, Eq)

///|
/// Phase-indexed compiler values keep later passes from consuming an
/// unselected or semantically unelaborated program.
pub struct SelectedAttentionTileCompilation {
  priv request : AttentionTileCompilerRequest
  priv plan : @luna_attention_strategy.CompiledAttentionTilePlan
} derive(Debug, Eq)

///|
pub struct SemanticAttentionTileCompilation {
  priv selected : SelectedAttentionTileCompilation
  priv program : @luna_attention_tile_ir.AttentionTileProgram
} derive(Debug, Eq)

///|
pub struct OptimizedAttentionTileCompilation {
  priv semantic : SemanticAttentionTileCompilation
  priv optimization : @luna_attention_tile_optimizer.AttentionTileOptimization
} derive(Debug, Eq)

///|
pub struct AttentionTileCompilationDigest(String) derive(Debug, Eq, Hash)

///|
pub struct ScheduledAttentionTileCompilation {
  priv optimized : OptimizedAttentionTileCompilation
  priv schedule : @luna_attention_tile_schedule.AttentionTileSchedule
  priv canonical : Bytes
  priv digest : AttentionTileCompilationDigest
} derive(Debug, Eq)

///|
pub(all) enum AttentionTileCompilerRule {
  StrategySelection
  SemanticElaboration
  FunctionalOptimization
  ScheduleDerivation
  ResourceFeedback
} derive(Debug, Eq)

///|
pub(all) suberror AttentionTileCompilerError {
  InvalidAttentionTileCompilation(AttentionTileCompilerRule)
} derive(Debug, Eq)
