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
/// Derives an immutable map/fold schedule without assigning vendor concepts.
pub fn schedule_attention_tile(
  semantic : SemanticAttentionTileCompilation,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  let schedule = @luna_attention_tile_schedule.schedule_attention_tile_program(
    semantic.program,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(ScheduleDerivation)
  }
  let candidate = semantic.selected.plan.candidate()
  let canonical = @utf8.encode(
    "luna-attention-functional-compilation-v1\n" +
    "candidate=" +
    candidate.stable_id().to_string() +
    "\n" +
    "program=" +
    semantic.program.digest().as_hex() +
    "\n" +
    "schedule=" +
    schedule.digest().as_hex() +
    "\n",
  )
  {
    semantic,
    schedule,
    canonical,
    digest: AttentionTileCompilationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}

///|
/// Total composition of the three pure compiler passes.
pub fn compile_attention_tile(
  request : AttentionTileCompilerRequest,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  schedule_attention_tile(
    elaborate_attention_tile(select_attention_tile(request)),
  )
}
