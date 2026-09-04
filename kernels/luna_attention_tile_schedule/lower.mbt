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
    AttentionAssociativeFold => 3
    AttentionVectorMap => 4
  }
}

///|
fn iteration_axis_code(axis : AttentionIterationAxis) -> Int {
  match axis {
    AttentionQueryRowAxis => 1
    AttentionQueryHeadAxis => 2
    AttentionKeyValuePartitionAxis => 3
    AttentionKeyValueAxis => 4
    AttentionSoftmaxPartitionAxis => 5
    AttentionHeadColumnAxis => 6
  }
}

///|
fn compute_code(compute : @luna_attention_strategy.AttentionTileCompute) -> Int {
  match compute {
    AttentionScalarDot => 1
    AttentionSubgroupDot => 2
    AttentionGroupedQuerySubgroupDot => 3
    AttentionGroupedQuerySplitSubgroupDot => 6
    AttentionMatrixDot => 4
    AttentionMatrixQkPv => 5
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
  key_value_partitions : Int,
) -> Unit raise AttentionTileScheduleError {
  if key_value_partitions <= 0 ||
    loops.length() < 2 ||
    loops[0].kind != AttentionParallelMap ||
    loops[0].axis != AttentionQueryRowAxis ||
    loops[1].kind != AttentionParallelMap ||
    loops[1].axis != AttentionQueryHeadAxis {
    raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
  if key_value_partitions == 1 {
    if loops.length() != 4 ||
      loops[2].kind != AttentionSequentialFold ||
      loops[2].axis != AttentionKeyValueAxis ||
      loops[3].kind != AttentionVectorMap ||
      loops[3].axis != AttentionHeadColumnAxis {
      raise InvalidAttentionTileSchedule(InvalidLoopNest)
    }
  } else if loops.length() != 6 ||
    loops[2].kind != AttentionParallelMap ||
    loops[2].axis != AttentionKeyValuePartitionAxis ||
    loops[3].kind != AttentionSequentialFold ||
    loops[3].axis != AttentionKeyValueAxis ||
    loops[4].kind != AttentionAssociativeFold ||
    loops[4].axis != AttentionSoftmaxPartitionAxis ||
    loops[5].kind != AttentionVectorMap ||
    loops[5].axis != AttentionHeadColumnAxis {
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
  key_value_partitions : Int,
  hoist_paged_key_value_page_lookup : Bool,
  reuse_online_softmax_tile_storage : Bool,
  shared_memory_bytes : Int64,
) -> Bytes {
  let candidate = program.candidate()
  let (subgroups, stages, vector_bytes) = candidate.execution_geometry()
  let builder = StringBuilder()
  builder <+ "luna-attention-tile-schedule-v2\n"
  builder <+ "program_sha256=\{program.digest().as_hex()}\n"
  builder <+ "kv_partitions=\{key_value_partitions}\n"
  builder <+
    "hoist_paged_key_value_page_lookup=\{hoist_paged_key_value_page_lookup}\n"
  builder <+
    "reuse_online_softmax_tile_storage=\{reuse_online_softmax_tile_storage}\n"
  for combinator in loops {
    builder <+
      "loop=\{iteration_kind_code(combinator.kind)},\{iteration_axis_code(combinator.axis)},\{combinator.extent},\{combinator.tile}\n"
  }
  builder <+ "compute=\{compute_code(candidate.compute())}\n"
  builder <+ "memory=\{memory_code(candidate.memory())}\n"
  builder <+
    "execution=\{subgroups},\{stages},\{vector_bytes},\{shared_memory_bytes}\n"
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
  schedule_attention_tile_program_with_partitions(program, 1, false, false)
}

///|
/// Schedules the optimizer's non-topological rewrites while retaining one
/// ordered KV fold. This is the ordinary single-kernel path; the partitioned
/// entry point below additionally materializes the associative split/merge.
pub fn schedule_rewritten_attention_tile_program(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  optimization : @luna_attention_tile_optimizer.AttentionTileOptimization,
) -> AttentionTileSchedule raise AttentionTileScheduleError {
  if optimization.program_digest() != program.digest() {
    raise InvalidAttentionTileSchedule(InvalidProblem)
  }
  schedule_attention_tile_program_with_partitions(
    program,
    1,
    attention_optimization_has_rule(optimization, HoistPagedKeyValuePageLookup),
    attention_optimization_has_rule(optimization, ReuseOnlineSoftmaxTileStorage),
  )
}

///|
/// Materializes the optimizer's pure partition/merge decision into explicit
/// map/fold combinators. Backends may map the partition map to blocks,
/// workgroups, threads, or CPU tasks without changing the merge semantics.
pub fn schedule_optimized_attention_tile_program(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  optimization : @luna_attention_tile_optimizer.AttentionTileOptimization,
) -> AttentionTileSchedule raise AttentionTileScheduleError {
  if optimization.program_digest() != program.digest() {
    raise InvalidAttentionTileSchedule(InvalidProblem)
  }
  let partitions = match optimization.reduction_topology() {
    AttentionOrderedOnlineSoftmaxFold => 1
    AttentionPartitionedOnlineSoftmaxFold(count) => count
  }
  schedule_attention_tile_program_with_partitions(
    program,
    partitions,
    attention_optimization_has_rule(optimization, HoistPagedKeyValuePageLookup),
    attention_optimization_has_rule(optimization, ReuseOnlineSoftmaxTileStorage),
  )
}

///|
fn attention_optimization_has_rule(
  optimization : @luna_attention_tile_optimizer.AttentionTileOptimization,
  expected : @luna_attention_tile_optimizer.AttentionTileRewriteRule,
) -> Bool {
  optimization.regions().any(region => region.rule() == expected)
}

///|
fn schedule_attention_tile_program_with_partitions(
  program : @luna_attention_tile_ir.AttentionTileProgram,
  key_value_partitions : Int,
  hoist_paged_key_value_page_lookup : Bool,
  reuse_online_softmax_tile_storage : Bool,
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
  let query_head_tile = match candidate.compute() {
    AttentionGroupedQuerySubgroupDot | AttentionGroupedQuerySplitSubgroupDot => {
      let (query_heads, key_value_heads, _, _) = problem.shape().dimensions()
      query_heads / key_value_heads
    }
    AttentionScalarDot
    | AttentionSubgroupDot
    | AttentionMatrixDot
    | AttentionMatrixQkPv => 1
  }
  let prefix : Array[AttentionLoopCombinator] = [
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
      tile: query_head_tile,
    },
  ]
  let suffix : Array[AttentionLoopCombinator] = if key_value_partitions == 1 {
    [
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
    ]
  } else {
    let partition_span = 1 + (context_tokens - 1) / key_value_partitions
    [
      {
        kind: AttentionParallelMap,
        axis: AttentionKeyValuePartitionAxis,
        extent: context_tokens,
        tile: partition_span,
      },
      {
        kind: AttentionSequentialFold,
        axis: AttentionKeyValueAxis,
        extent: partition_span,
        tile: key_value_tile,
      },
      {
        kind: AttentionAssociativeFold,
        axis: AttentionSoftmaxPartitionAxis,
        extent: key_value_partitions,
        tile: 1,
      },
      {
        kind: AttentionVectorMap,
        axis: AttentionHeadColumnAxis,
        extent: head_dimension,
        tile: vector_elements,
      },
    ]
  }
  for combinator in suffix {
    prefix.push(combinator)
  }
  let loops = ReadOnlyArray::from_array(prefix)
  validate_schedule_loops(loops, key_value_partitions)
  let shared_memory_bytes = if reuse_online_softmax_tile_storage &&
    candidate.compute() == AttentionMatrixQkPv {
    candidate.shared_memory_bytes() -
    query_tile.to_int64() * key_value_tile.to_int64() * 2L -
    query_tile.to_int64() * 12L -
    16L
  } else {
    candidate.shared_memory_bytes()
  }
  let canonical = canonical_schedule(
    program, loops, key_value_partitions, hoist_paged_key_value_page_lookup, reuse_online_softmax_tile_storage,
    shared_memory_bytes,
  )
  {
    program_digest: program.digest(),
    loops,
    compute: candidate.compute(),
    memory: candidate.memory(),
    subgroups_per_block,
    pipeline_stages,
    vector_bytes,
    shared_memory_bytes,
    key_value_partitions,
    hoist_paged_key_value_page_lookup,
    reuse_online_softmax_tile_storage,
    canonical,
    digest: AttentionTileScheduleDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
