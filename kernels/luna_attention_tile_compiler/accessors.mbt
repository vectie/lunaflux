///|
pub fn AttentionTileCompilerRequest::problem(
  self : AttentionTileCompilerRequest,
) -> @luna_attention_strategy.AttentionCompileProblem {
  self.problem
}

///|
pub fn SelectedAttentionTileCompilation::plan(
  self : SelectedAttentionTileCompilation,
) -> @luna_attention_strategy.CompiledAttentionTilePlan {
  self.plan
}

///|
pub fn SemanticAttentionTileCompilation::plan(
  self : SemanticAttentionTileCompilation,
) -> @luna_attention_strategy.CompiledAttentionTilePlan {
  self.selected.plan
}

///|
pub fn SemanticAttentionTileCompilation::program(
  self : SemanticAttentionTileCompilation,
) -> @luna_attention_tile_ir.AttentionTileProgram {
  self.program
}

///|
pub fn OptimizedAttentionTileCompilation::plan(
  self : OptimizedAttentionTileCompilation,
) -> @luna_attention_strategy.CompiledAttentionTilePlan {
  self.semantic.selected.plan
}

///|
pub fn OptimizedAttentionTileCompilation::program(
  self : OptimizedAttentionTileCompilation,
) -> @luna_attention_tile_ir.AttentionTileProgram {
  self.semantic.program
}

///|
pub fn OptimizedAttentionTileCompilation::optimization(
  self : OptimizedAttentionTileCompilation,
) -> @luna_attention_tile_optimizer.AttentionTileOptimization {
  self.optimization
}

///|
pub fn ScheduledAttentionTileCompilation::plan(
  self : ScheduledAttentionTileCompilation,
) -> @luna_attention_strategy.CompiledAttentionTilePlan {
  self.optimized.semantic.selected.plan
}

///|
pub fn ScheduledAttentionTileCompilation::program(
  self : ScheduledAttentionTileCompilation,
) -> @luna_attention_tile_ir.AttentionTileProgram {
  self.optimized.semantic.program
}

///|
pub fn ScheduledAttentionTileCompilation::optimization(
  self : ScheduledAttentionTileCompilation,
) -> @luna_attention_tile_optimizer.AttentionTileOptimization {
  self.optimized.optimization
}

///|
pub fn ScheduledAttentionTileCompilation::schedule(
  self : ScheduledAttentionTileCompilation,
) -> @luna_attention_tile_schedule.AttentionTileSchedule {
  self.schedule
}

///|
pub fn ScheduledAttentionTileCompilation::canonical_bytes(
  self : ScheduledAttentionTileCompilation,
) -> Bytes {
  self.canonical
}

///|
pub fn ScheduledAttentionTileCompilation::digest(
  self : ScheduledAttentionTileCompilation,
) -> AttentionTileCompilationDigest {
  self.digest
}

///|
pub fn AttentionTileCompilationFrontier::selected(
  self : AttentionTileCompilationFrontier,
) -> ScheduledAttentionTileCompilation {
  self.selected
}

///|
pub fn AttentionTileCompilationFrontier::variants(
  self : AttentionTileCompilationFrontier,
) -> ReadOnlyArray[ScheduledAttentionTileCompilation] {
  self.variants
}

///|
pub fn AttentionTileCompilationFrontier::canonical_bytes(
  self : AttentionTileCompilationFrontier,
) -> Bytes {
  self.canonical
}

///|
pub fn AttentionTileCompilationFrontier::digest(
  self : AttentionTileCompilationFrontier,
) -> AttentionTileCompilationDigest {
  self.digest
}

///|
pub fn AttentionTileCompilationDigest::as_hex(
  self : AttentionTileCompilationDigest,
) -> String {
  self.0
}
