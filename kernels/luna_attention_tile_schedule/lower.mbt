///|
fn numeric_bytes(
  numeric_type : @luna_attention_strategy.AttentionNumericType,
) -> Int {
  match numeric_type {
    AttentionBFloat16 | AttentionFloat16 => 2
    AttentionFloat32 => 4
    AttentionFloat8 => 1
  }
}

///|
fn iteration_kind_code(kind : AttentionIterationKind) -> Int {
  match kind {
    AttentionParallelMap => 1
    AttentionSequentialFold => 2
    AttentionVectorMap => 3
  }
}

///|
fn iteration_axis_code(axis : AttentionIterationAxis) -> Int {
  match axis {
    AttentionQueryRowAxis => 1
    AttentionQueryHeadAxis => 2
    AttentionKeyValueAxis => 3
    AttentionHeadColumnAxis => 4
  }
}

///|
fn compute_code(compute : @luna_attention_strategy.AttentionTileCompute) -> Int {
  match compute {
    AttentionScalarDot => 1
    AttentionSubgroupDot => 2
    AttentionMatrixDot => 3
  }
}

///|
fn memory_code(memory : @luna_attention_strategy.AttentionTileMemory) -> Int {
  match memory {
    AttentionDirectGlobal => 1
    AttentionStagedShared => 2
    AttentionAsyncStagedShared => 3
  }
}

///|
fn validate_schedule_loops(
  loops : ReadOnlyArray[AttentionLoopCombinator],
) -> Unit raise AttentionTileScheduleError {
  if loops.length() != 4 ||
    loops[0].kind != AttentionParallelMap ||
    loops[0].axis != AttentionQueryRowAxis ||
    loops[1].kind != AttentionParallelMap ||
    loops[1].axis != AttentionQueryHeadAxis ||
    loops[2].kind != AttentionSequentialFold ||
    loops[2].axis != AttentionKeyValueAxis ||
    loops[3].kind != AttentionVectorMap ||
    loops[3].axis != AttentionHeadColumnAxis {
    raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
  for combinator in loops {
    if combinator.extent <= 0 || combinator.tile <= 0 {
      raise InvalidAttentionTileSchedule(InvalidLoopNest)
    }
  }
}

///|
fn canonical_schedule(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  loops : ReadOnlyArray[AttentionLoopCombinator],
) -> Bytes {
  let candidate = program.candidate()
  let (subgroups, stages, vector_bytes) = candidate.execution_geometry()
  let builder = StringBuilder()
  builder <+ "luna-attention-tile-schedule-v1\n"
  builder <+ "program_sha256=\{program.digest().as_hex()}\n"
  for combinator in loops {
    builder <+
      "loop=\{iteration_kind_code(combinator.kind)},\{iteration_axis_code(combinator.axis)},\{combinator.extent},\{combinator.tile}\n"
  }
  builder <+ "compute=\{compute_code(candidate.compute())}\n"
  builder <+ "memory=\{memory_code(candidate.memory())}\n"
  builder <+
    "execution=\{subgroups},\{stages},\{vector_bytes},\{candidate.shared_memory_bytes()}\n"
  @utf8.encode(builder.to_string())
}

///|
/// Schedules semantic attention as nested pure map/fold combinators. Query
/// rows and heads are parallel maps, paged K/V traversal is the sole ordered
/// fold, and head components form a vector map. This boundary makes reduction
/// order explicit before any backend chooses concrete threads or instructions.
pub fn schedule_attention_tile_program(
  program : @luna_attention_tile_ir.AttentionTileProgram,
) -> AttentionTileSchedule raise AttentionTileScheduleError {
  let problem = program.problem()
  let candidate = program.candidate()
  let (query_rows, context_tokens) = problem.dimensions()
  let (query_heads, _, head_dimension, _) = problem.shape().dimensions()
  let (query_tile, key_value_tile, tile_head) = candidate.tile().dimensions()
  let (subgroups_per_block, pipeline_stages, vector_bytes) = candidate.execution_geometry()
  if tile_head != head_dimension {
    raise InvalidAttentionTileSchedule(InvalidTile)
  }
  let element_bytes = numeric_bytes(problem.numeric_type())
  if vector_bytes < element_bytes || vector_bytes % element_bytes != 0 {
    raise InvalidAttentionTileSchedule(InvalidVector)
  }
  let vector_elements = vector_bytes / element_bytes
  if head_dimension % vector_elements != 0 {
    raise InvalidAttentionTileSchedule(InvalidVector)
  }
  let loops = ReadOnlyArray::from_array([
    {
      kind: AttentionParallelMap,
      axis: AttentionQueryRowAxis,
      extent: query_rows,
      tile: query_tile,
    },
    {
      kind: AttentionParallelMap,
      axis: AttentionQueryHeadAxis,
      extent: query_heads,
      tile: 1,
    },
    {
      kind: AttentionSequentialFold,
      axis: AttentionKeyValueAxis,
      extent: context_tokens,
      tile: key_value_tile,
    },
    {
      kind: AttentionVectorMap,
      axis: AttentionHeadColumnAxis,
      extent: head_dimension,
      tile: vector_elements,
    },
  ])
  validate_schedule_loops(loops)
  let canonical = canonical_schedule(program, loops)
  let shared_memory_bytes = candidate.shared_memory_bytes()
  {
    program_digest: program.digest(),
    loops,
    compute: candidate.compute(),
    memory: candidate.memory(),
    subgroups_per_block,
    pipeline_stages,
    vector_bytes,
    shared_memory_bytes,
    canonical,
    digest: AttentionTileScheduleDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
