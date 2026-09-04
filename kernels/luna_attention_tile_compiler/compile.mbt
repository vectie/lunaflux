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
    parallel_workgroup_target=semantic.selected.request.capabilities.parallel_workgroup_target(),
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
/// Conservative post-rewrite throughput score. The compiler only credits the
/// first shared-memory occupancy knee: proving two resident workgroups is
/// useful on every parallel backend, while claiming higher residency would
/// require register-file and execution-unit limits not present in the portable
/// capability contract.
fn attention_tile_resource_score(
  request : AttentionTileCompilerRequest,
  scheduled : ScheduledAttentionTileCompilation,
) -> Int64 raise AttentionTileCompilerError {
  let base = scheduled.plan().static_cost()
  let (_, shared_capacity, _) = request.capabilities.limits()
  let (_, _, _, shared_bytes) = scheduled.schedule().execution_geometry()
  let loops = scheduled.schedule().loops()
  if base <= 0L || loops.length() < 2 {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let (query_extent, query_tile) = loops[0].geometry()
  let (head_extent, head_tile) = loops[1].geometry()
  let logical_workgroups = (1 + (query_extent - 1) / query_tile) *
    (1 + (head_extent - 1) / head_tile) *
    request.problem.batch_rows()
  let shared_residency = if shared_bytes <= 0L {
    1
  } else {
    Int::max(1, Int::min(2, (shared_capacity / shared_bytes).to_int()))
  }
  let residency = Int::max(1, Int::min(logical_workgroups, shared_residency))
  let divisor = residency.to_int64()
  base / divisor + (if base % divisor == 0L { 0L } else { 1L })
}

///|
fn compile_static_attention_tile_candidate(
  request : AttentionTileCompilerRequest,
  candidate : @luna_attention_strategy.AttentionTileCandidate,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  let plan = @luna_attention_strategy.compile_attention_tile_candidate_plan(
    request.problem,
    request.capabilities,
    candidate,
  ) catch {
    _ => raise InvalidAttentionTileCompilation(StrategySelection)
  }
  schedule_attention_tile(
    optimize_attention_tile(elaborate_attention_tile({ request, plan })),
  )
}

///|
/// True when `left` is never more expensive and never larger than `right`,
/// with at least one strict improvement. Both metrics are compiler values:
/// static semantic work and post-rewrite peak shared storage. Vendor-specific
/// occupancy or instruction names are deliberately absent.
fn attention_tile_dominates(
  left : ScheduledAttentionTileCompilation,
  right : ScheduledAttentionTileCompilation,
) -> Bool {
  let left_cost = left.plan().static_cost()
  let right_cost = right.plan().static_cost()
  let (_, _, _, left_shared) = left.schedule().execution_geometry()
  let (_, _, _, right_shared) = right.schedule().execution_geometry()
  left_cost <= right_cost &&
  left_shared <= right_shared &&
  (left_cost < right_cost || left_shared < right_shared)
}

///|
/// Compiles the non-dominated kernel family for one semantic problem. The
/// result is a pure Pareto frontier rather than a guessed runtime winner:
/// execution-graph buckets can later select a throughput- or residency-biased
/// member without rebuilding the compiler pipeline. Exact offline autotune is
/// authoritative and therefore collapses the frontier to its measured winner.
pub fn compile_attention_tile_frontier(
  request : AttentionTileCompilerRequest,
) -> AttentionTileCompilationFrontier raise AttentionTileCompilerError {
  let selected = select_attention_tile(request)
  let variants : ReadOnlyArray[ScheduledAttentionTileCompilation] = if selected
    .plan()
    .source() ==
    AttentionOfflineAutotune {
    ReadOnlyArray::from_array([
      schedule_attention_tile(
        optimize_attention_tile(elaborate_attention_tile(selected)),
      ),
    ])
  } else {
    let candidates = @luna_attention_strategy.generate_attention_tile_candidates(
      request.problem,
      request.capabilities,
    ) catch {
      _ => raise InvalidAttentionTileCompilation(StrategySelection)
    }
    let compiled = candidates.map(candidate => {
      compile_static_attention_tile_candidate(request, candidate)
    })
    compiled.filter(candidate => {
      !compiled.any(other => {
        other.plan().candidate().stable_id() !=
        candidate.plan().candidate().stable_id() &&
        attention_tile_dominates(other, candidate)
      })
    })
  }
  if variants.length() == 0 {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let resource_selected = variants[1:].fold(init=variants[0], (
    current,
    candidate,
  ) => {
    let current_score = attention_tile_resource_score(request, current)
    let candidate_score = attention_tile_resource_score(request, candidate)
    if candidate_score < current_score ||
      (
        candidate_score == current_score &&
        candidate.plan().candidate().stable_id() <
        current.plan().candidate().stable_id()
      ) {
      candidate
    } else {
      current
    }
  })
  let builder = StringBuilder()
  builder <+ "luna-attention-functional-frontier-v1\n"
  builder <+ "selected=\{resource_selected.digest().as_hex()}\n"
  builder <+ "variants=\{variants.length()}\n"
  for variant in variants {
    let (_, _, _, shared) = variant.schedule().execution_geometry()
    builder <+
      "variant=\{variant.plan().candidate().stable_id()},\{variant.plan().static_cost()},\{shared},\{variant.digest().as_hex()}\n"
  }
  let canonical = @utf8.encode(builder.to_string())
  {
    selected: resource_selected,
    variants,
    canonical,
    digest: AttentionTileCompilationDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}

///|
/// Total composition of the pure compiler passes. Immutable offline autotune
/// observations retain authority. Without an exact observation, every bounded
/// candidate is elaborated, optimized, and scheduled, then the optimized peak
/// storage feeds back into the final portable strategy choice.
pub fn compile_attention_tile(
  request : AttentionTileCompilerRequest,
) -> ScheduledAttentionTileCompilation raise AttentionTileCompilerError {
  compile_attention_tile_frontier(request).selected()
}
