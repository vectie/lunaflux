///|
pub extend AttentionResourceBudget with @debug.Debug::{to_repr}

///|
pub extend AttentionResourceBudget with Eq::{not_equal, equal}

///|
pub extend AttentionRegisterObservation with @debug.Debug::{to_repr}

///|
pub extend AttentionRegisterObservation with Eq::{not_equal, equal}

///|
pub extend AttentionTileCompilerRequest with @debug.Debug::{to_repr}

///|
pub extend AttentionTileCompilerRequest with Eq::{not_equal, equal}

///|
pub extend SelectedAttentionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend SelectedAttentionTileCompilation with Eq::{not_equal, equal}

///|
pub extend SemanticAttentionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend SemanticAttentionTileCompilation with Eq::{not_equal, equal}

///|
pub extend OptimizedAttentionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend OptimizedAttentionTileCompilation with Eq::{not_equal, equal}

///|
pub extend AttentionTileCompilationDigest with @debug.Debug::{to_repr}

///|
pub extend AttentionTileCompilationDigest with Eq::{not_equal, equal}

///|
pub extend AttentionTileCompilationDigest with Hash::{hash, hash_combine}

///|
pub extend ScheduledAttentionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend ScheduledAttentionTileCompilation with Eq::{not_equal, equal}

///|
pub extend AttentionTileCompilationFrontier with @debug.Debug::{to_repr}

///|
pub extend AttentionTileCompilationFrontier with Eq::{not_equal, equal}

///|
pub extend AttentionPartitionedTileCompilationVariant with @debug.Debug::{
  to_repr,
}

///|
pub extend AttentionPartitionedTileCompilationVariant with Eq::{
  not_equal,
  equal,
}

///|
pub extend AttentionPartitionedTileCompilationFrontier with @debug.Debug::{
  to_repr,
}

///|
pub extend AttentionPartitionedTileCompilationFrontier with Eq::{
  not_equal,
  equal,
}

///|
pub extend AttentionTileCompilerRule with @debug.Debug::{to_repr}

///|
pub extend AttentionTileCompilerRule with Eq::{not_equal, equal}

///|
pub extend AttentionTileCompilerError with @debug.Debug::{to_repr}

///|
pub extend AttentionTileCompilerError with Eq::{not_equal, equal}
