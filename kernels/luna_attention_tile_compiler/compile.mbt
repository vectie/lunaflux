///|
/// Pure strategy selection. Offline observations are values supplied by the
/// caller rather than an implicit mutable autotune database.
pub fn select_attention_tile(
  request : AttentionTileCompilerRequest,
) -> SelectedAttentionTileCompilation raise AttentionTileCompilerError {
  let plan = @luna_attention_strategy.compile_attention_tile_plan(
    request.autotune_records,
    backend=request.backend,
    target=request.target,
    problem=request.problem,
    capabilities=request.capabilities,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(StrategySelection)
  }
  { request, plan }
}

///|
/// Elaborates the selected strategy into the backend-neutral semantic IR.
pub fn elaborate_attention_tile(
  selected : SelectedAttentionTileCompilation,
) -> SemanticAttentionTileCompilation raise AttentionTileCompilerError {
  let program = @luna_attention_tile_ir.compile_attention_tile_ir(selected.plan) catch {
    _ => raise InvalidAttentionTileCompilation(SemanticElaboration)
  }
  { selected, program }
}

///|
/// Applies the backend-neutral functional rewrite pipeline.
pub fn optimize_attention_tile(
  semantic : SemanticAttentionTileCompilation,
) -> OptimizedAttentionTileCompilation raise AttentionTileCompilerError {
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    semantic.program,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(FunctionalOptimization)
  }
  { semantic, optimization }
}

///|
/// Derives an immutable map/fold schedule without assigning vendor concepts.
pub fn schedule_attention_tile(
  optimized : OptimizedAttentionTileCompilation,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  let schedule = @luna_attention_tile_schedule.schedule_rewritten_attention_tile_program(
    optimized.semantic.program,
    optimized.optimization,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(ScheduleDerivation)
  }
  finish_attention_tile_schedule(optimized, schedule)
}

///|
/// Schedules the optimizer-selected partition-map/associative-merge topology.
/// It is a separate typed entry point so a backend cannot accidentally consume
/// a partitioned program through a legacy single-launch emitter.
pub fn schedule_partitioned_attention_tile(
  optimized : OptimizedAttentionTileCompilation,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  let schedule = @luna_attention_tile_schedule.schedule_optimized_attention_tile_program(
    optimized.semantic.program,
    optimized.optimization,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(ScheduleDerivation)
  }
  if schedule.key_value_partitions() <= 1 {
    raise InvalidAttentionTileCompilation(ScheduleDerivation)
  }
  finish_attention_tile_schedule(optimized, schedule)
}

///|
fn finish_attention_tile_schedule(
  optimized : OptimizedAttentionTileCompilation,
  schedule : @luna_attention_tile_schedule.AttentionTileSchedule,
) -> ScheduledAttentionTileCompilation {
  let candidate = optimized.semantic.selected.plan.candidate()
  let canonical = @utf8.encode(
    "luna-attention-functional-compilation-v1\n" +
    "candidate=" +
    candidate.stable_id().to_string() +
    "\n" +
    "program=" +
    optimized.semantic.program.digest().as_hex() +
    "\n" +
    "optimization=" +
    optimized.optimization.digest().as_hex() +
    "\n" +
    "schedule=" +
    schedule.digest().as_hex() +
    "\n",
  )
  {
    optimized,
    schedule,
    canonical,
    digest: AttentionTileCompilationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}

///|
/// Total pure composition for a backend that supports multi-launch
/// partitioned prefill attention.
pub fn compile_partitioned_attention_tile(
  request : AttentionTileCompilerRequest,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  schedule_partitioned_attention_tile(
    optimize_attention_tile(
      elaborate_attention_tile(select_attention_tile(request)),
    ),
  )
}

///|
/// Total composition of the four pure compiler passes.
pub fn compile_attention_tile(
  request : AttentionTileCompilerRequest,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  schedule_attention_tile(
    optimize_attention_tile(
      elaborate_attention_tile(select_attention_tile(request)),
    ),
  )
}
