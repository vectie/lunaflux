///|
fn attention_numeric_bytes(numeric_type : AttentionNumericType) -> Int64 {
  match numeric_type {
    AttentionBFloat16 | AttentionFloat16 => 2L
    AttentionFloat32 => 4L
    AttentionFloat8 => 1L
  }
}

///|
fn attention_checked_add(
  left : Int64,
  right : Int64,
  rule : AttentionStrategyRule,
) -> Int64 raise AttentionStrategyError {
  if left < 0L || right < 0L || left > 9223372036854775807L - right {
    raise InvalidAttentionStrategy(rule)
  }
  left + right
}

///|
fn attention_ceil_div(value : Int, divisor : Int) -> Int {
  1 + (value - 1) / divisor
}

///|
fn attention_candidate_shared_bytes(
  problem : AttentionCompileProblem,
  query_tile : Int,
  key_value_tile : Int,
  pipeline_stages : Int,
  memory : AttentionTileMemory,
) -> Int64 raise AttentionStrategyError {
  if memory == AttentionDirectGlobal {
    return 0L
  }
  let (_, _, head_dimension, _) = problem.shape.dimensions()
  let element_bytes = attention_numeric_bytes(problem.numeric_type)
  let query_elements = attention_checked_product(
    query_tile.to_int64(),
    head_dimension.to_int64(),
    Candidate,
  )
  let key_value_elements = attention_checked_product(
    attention_checked_product(
      key_value_tile.to_int64(),
      head_dimension.to_int64(),
      Candidate,
    ),
    2L,
    Candidate,
  )
  let staged_key_value_elements = attention_checked_product(
    key_value_elements,
    pipeline_stages.to_int64(),
    Candidate,
  )
  let operand_elements = attention_checked_add(
    query_elements,
    staged_key_value_elements,
    Candidate,
  )
  let operand_bytes = attention_checked_product(
    operand_elements,
    element_bytes,
    Candidate,
  )
  let accumulator_bytes = attention_checked_product(
    query_elements,
    4L,
    Candidate,
  )
  let score_elements = attention_checked_product(
    query_tile.to_int64(),
    key_value_tile.to_int64(),
    Candidate,
  )
  let score_and_weight_bytes = attention_checked_product(
    score_elements,
    8L,
    Candidate,
  )
  let online_state_bytes = attention_checked_product(
    query_tile.to_int64(),
    8L,
    Candidate,
  )
  attention_checked_add(
    attention_checked_add(
      attention_checked_add(operand_bytes, accumulator_bytes, Candidate),
      score_and_weight_bytes,
      Candidate,
    ),
    online_state_bytes,
    Candidate,
  )
}

///|
fn attention_candidate_compatible(
  problem : AttentionCompileProblem,
  capabilities : AttentionCompilerCapabilities,
  candidate : AttentionTileCandidate,
) -> Bool {
  let (_, _, head_dimension, _) = problem.shape.dimensions()
  let (query_rows, key_value_rows, head_columns) = candidate.tile.dimensions()
  let (subgroups, stages, vector_bytes) = candidate.execution_geometry()
  if query_rows <= 0 ||
    key_value_rows <= 0 ||
    head_columns != head_dimension ||
    subgroups * capabilities.subgroup_width > capabilities.max_threads_per_block ||
    stages > capabilities.max_pipeline_stages ||
    candidate.shared_memory_bytes > capabilities.max_shared_memory_bytes ||
    head_dimension *
    attention_numeric_bytes(problem.numeric_type).to_int() %
    vector_bytes !=
    0 {
    return false
  }
  if candidate.memory == AttentionAsyncStagedShared &&
    !capabilities.supports_async_copy {
    return false
  }
  match candidate.compute {
    AttentionMatrixDot =>
      match capabilities.matrix_shape {
        None => false
        Some(matrix) => {
          let (rows, columns, reduction) = matrix.dimensions()
          query_rows % rows == 0 &&
          key_value_rows % columns == 0 &&
          head_dimension % reduction == 0
        }
      }
    AttentionScalarDot | AttentionSubgroupDot => true
  }
}

///|
fn make_attention_candidate(
  problem : AttentionCompileProblem,
  capabilities : AttentionCompilerCapabilities,
  stable_id : Int,
  compute : AttentionTileCompute,
  memory : AttentionTileMemory,
  query_rows : Int,
  key_value_rows : Int,
  subgroups_per_block : Int,
  pipeline_stages : Int,
  vector_bytes : Int,
) -> AttentionTileCandidate? raise AttentionStrategyError {
  let (_, _, head_columns, _) = problem.shape.dimensions()
  let candidate = {
    stable_id,
    compute,
    memory,
    tile: { query_rows, key_value_rows, head_columns },
    subgroups_per_block,
    pipeline_stages,
    vector_bytes,
    shared_memory_bytes: attention_candidate_shared_bytes(
      problem, query_rows, key_value_rows, pipeline_stages, memory,
    ),
  }
  if attention_candidate_compatible(problem, capabilities, candidate) {
    Some(candidate)
  } else {
    None
  }
}

///|
priv struct AttentionCandidateSpec {
  stable_id : Int
  phase : AttentionExecutionPhase
  compute : AttentionTileCompute
  memory : AttentionTileMemory
  query_rows : Int
  key_value_rows : Int
  subgroups_per_block : Int
  pipeline_stages : Int
  vector_bytes : Int
  minimum_query_rows : Int
  minimum_context_tokens : Int
}

///|
fn attention_candidate_specs() -> Array[AttentionCandidateSpec] {
  [
    {
      stable_id: 100,
      phase: Prefill,
      compute: AttentionScalarDot,
      memory: AttentionDirectGlobal,
      query_rows: 1,
      key_value_rows: 16,
      subgroups_per_block: 1,
      pipeline_stages: 1,
      vector_bytes: 4,
      minimum_query_rows: 1,
      minimum_context_tokens: 1,
    },
    {
      stable_id: 200,
      phase: Prefill,
      compute: AttentionSubgroupDot,
      memory: AttentionStagedShared,
      query_rows: 4,
      key_value_rows: 32,
      subgroups_per_block: 4,
      pipeline_stages: 2,
      vector_bytes: 8,
      minimum_query_rows: 4,
      minimum_context_tokens: 16,
    },
    {
      stable_id: 210,
      phase: Prefill,
      compute: AttentionSubgroupDot,
      memory: AttentionStagedShared,
      query_rows: 8,
      key_value_rows: 64,
      subgroups_per_block: 4,
      pipeline_stages: 2,
      vector_bytes: 8,
      minimum_query_rows: 8,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 211,
      phase: Prefill,
      compute: AttentionSubgroupDot,
      memory: AttentionAsyncStagedShared,
      query_rows: 8,
      key_value_rows: 64,
      subgroups_per_block: 4,
      pipeline_stages: 3,
      vector_bytes: 16,
      minimum_query_rows: 8,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 220,
      phase: Prefill,
      compute: AttentionSubgroupDot,
      memory: AttentionStagedShared,
      query_rows: 16,
      key_value_rows: 64,
      subgroups_per_block: 8,
      pipeline_stages: 2,
      vector_bytes: 8,
      minimum_query_rows: 16,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 221,
      phase: Prefill,
      compute: AttentionSubgroupDot,
      memory: AttentionAsyncStagedShared,
      query_rows: 16,
      key_value_rows: 128,
      subgroups_per_block: 8,
      pipeline_stages: 3,
      vector_bytes: 16,
      minimum_query_rows: 16,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 300,
      phase: Prefill,
      compute: AttentionMatrixDot,
      memory: AttentionStagedShared,
      query_rows: 16,
      key_value_rows: 64,
      subgroups_per_block: 8,
      pipeline_stages: 2,
      vector_bytes: 16,
      minimum_query_rows: 16,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 301,
      phase: Prefill,
      compute: AttentionMatrixDot,
      memory: AttentionAsyncStagedShared,
      query_rows: 16,
      key_value_rows: 128,
      subgroups_per_block: 8,
      pipeline_stages: 3,
      vector_bytes: 16,
      minimum_query_rows: 16,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 400,
      phase: Decode,
      compute: AttentionScalarDot,
      memory: AttentionDirectGlobal,
      query_rows: 1,
      key_value_rows: 16,
      subgroups_per_block: 1,
      pipeline_stages: 1,
      vector_bytes: 4,
      minimum_query_rows: 1,
      minimum_context_tokens: 1,
    },
    {
      stable_id: 410,
      phase: Decode,
      compute: AttentionSubgroupDot,
      memory: AttentionStagedShared,
      query_rows: 1,
      key_value_rows: 64,
      subgroups_per_block: 4,
      pipeline_stages: 2,
      vector_bytes: 8,
      minimum_query_rows: 1,
      minimum_context_tokens: 32,
    },
    {
      stable_id: 411,
      phase: Decode,
      compute: AttentionSubgroupDot,
      memory: AttentionAsyncStagedShared,
      query_rows: 1,
      key_value_rows: 128,
      subgroups_per_block: 4,
      pipeline_stages: 3,
      vector_bytes: 16,
      minimum_query_rows: 1,
      minimum_context_tokens: 32,
    },
  ]
}

///|
/// Generates a deterministic, bounded candidate set. This is compiler/startup
/// work; request-time execution consumes only the selected plan.
pub fn generate_attention_tile_candidates(
  problem : AttentionCompileProblem,
  capabilities : AttentionCompilerCapabilities,
) -> ReadOnlyArray[AttentionTileCandidate] raise AttentionStrategyError {
  let (query_rows, context_tokens) = problem.dimensions()
  let candidates = attention_candidate_specs().filter_map(spec => {
    if spec.phase != problem.phase ||
      query_rows < spec.minimum_query_rows ||
      context_tokens < spec.minimum_context_tokens {
      None
    } else {
      make_attention_candidate(
        problem,
        capabilities,
        spec.stable_id,
        spec.compute,
        spec.memory,
        spec.query_rows,
        spec.key_value_rows,
        spec.subgroups_per_block,
        spec.pipeline_stages,
        spec.vector_bytes,
      )
    }
  })
  if candidates.length() == 0 {
    raise InvalidAttentionStrategy(Capability)
  }
  ReadOnlyArray::from_array(candidates)
}

///|
fn attention_candidate_static_cost(
  problem : AttentionCompileProblem,
  candidate : AttentionTileCandidate,
) -> Int64 raise AttentionStrategyError {
  let (query_rows, context_tokens) = problem.dimensions()
  let (query_tile, key_value_tile, head_columns) = candidate.tile.dimensions()
  let query_tiles = attention_ceil_div(query_rows, query_tile).to_int64()
  let key_value_tiles = attention_ceil_div(context_tokens, key_value_tile).to_int64()
  let padded_queries = attention_checked_product(
    query_tiles,
    query_tile.to_int64(),
    Candidate,
  )
  let padded_keys = attention_checked_product(
    key_value_tiles,
    key_value_tile.to_int64(),
    Candidate,
  )
  let dot_work = attention_checked_product(
    attention_checked_product(padded_queries, padded_keys, Candidate),
    head_columns.to_int64(),
    Candidate,
  )
  let arithmetic_divisor = match candidate.compute {
    AttentionScalarDot => 1L
    AttentionSubgroupDot => candidate.subgroups_per_block.to_int64() * 8L
    AttentionMatrixDot => candidate.subgroups_per_block.to_int64() * 64L
  }
  let arithmetic_cost = dot_work / arithmetic_divisor
  let key_value_bytes = attention_checked_product(
    attention_checked_product(
      attention_checked_product(query_tiles, padded_keys, Candidate),
      head_columns.to_int64(),
      Candidate,
    ),
    attention_numeric_bytes(problem.numeric_type) * 2L,
    Candidate,
  )
  let memory_multiplier = match candidate.memory {
    AttentionDirectGlobal => 4L
    AttentionStagedShared => 2L
    AttentionAsyncStagedShared => 1L
  }
  let memory_cost = attention_checked_product(
    key_value_bytes,
    memory_multiplier,
    Candidate,
  )
  let (query_heads, _, _, _) = problem.shape.dimensions()
  let scheduling_cost = attention_checked_product(
    attention_checked_product(query_tiles, query_heads.to_int64(), Candidate),
    4096L,
    Candidate,
  )
  attention_checked_add(
    attention_checked_add(arithmetic_cost, memory_cost, Candidate),
    scheduling_cost,
    Candidate,
  )
}

///|
fn validate_attention_autotune_records(
  records : ArrayView[AttentionAutotuneRecord],
) -> Unit raise AttentionStrategyError {
  for left_index, left in records {
    let mut right_index = left_index + 1
    while right_index < records.length() {
      let right = records[right_index]
      if left.key == right.key {
        if left.median_latency_ns == right.median_latency_ns &&
          left.sample_count == right.sample_count {
          raise InvalidAttentionStrategy(Duplicate)
        }
        raise InvalidAttentionStrategy(Conflict)
      }
      let left_key = left.key
      let right_key = right.key
      if left_key.backend == right_key.backend &&
        left_key.target == right_key.target &&
        left_key.problem == right_key.problem &&
        left_key.candidate.stable_id == right_key.candidate.stable_id {
        raise InvalidAttentionStrategy(Conflict)
      }
      right_index += 1
    }
  }
}

///|
/// Compiles one serving bucket. Exact matching offline measurements override
/// the deterministic cost model; ties are resolved by stable candidate ID.
pub fn compile_attention_tile_plan(
  records : ArrayView[AttentionAutotuneRecord],
  backend~ : AttentionBackendId,
  target~ : AttentionTargetId,
  problem~ : AttentionCompileProblem,
  capabilities~ : AttentionCompilerCapabilities,
) -> CompiledAttentionTilePlan raise AttentionStrategyError {
  validate_attention_autotune_records(records)
  let candidates = generate_attention_tile_candidates(problem, capabilities)
  let initial = candidates[0]
  let (selected, selected_cost) = candidates[1:].fold(
    init=(initial, attention_candidate_static_cost(problem, initial)),
    (current, candidate) => {
      let (current_candidate, current_cost) = current
      let cost = attention_candidate_static_cost(problem, candidate)
      if cost < current_cost ||
        (
          cost == current_cost &&
          candidate.stable_id < current_candidate.stable_id
        ) {
        (candidate, cost)
      } else {
        current
      }
    },
  )
  let selected_record = records.fold(init=None, (
    current : AttentionAutotuneRecord?,
    record,
  ) => {
    let key = record.key
    if key.backend == backend &&
      key.target == target &&
      key.problem == problem &&
      attention_candidate_compatible(problem, capabilities, key.candidate) {
      let candidate_present = candidates.any(candidate => {
        candidate == key.candidate
      })
      if candidate_present {
        match current {
          None => Some(record)
          Some(selected) =>
            if record.median_latency_ns < selected.median_latency_ns ||
              (
                record.median_latency_ns == selected.median_latency_ns &&
                record.key.candidate.stable_id <
                selected.key.candidate.stable_id
              ) {
              Some(record)
            } else {
              current
            }
        }
      } else {
        current
      }
    } else {
      current
    }
  })
  match selected_record {
    Some(record) =>
      {
        problem,
        candidate: record.key.candidate,
        source: AttentionOfflineAutotune,
        static_cost: attention_candidate_static_cost(
          problem,
          record.key.candidate,
        ),
      }
    None =>
      {
        problem,
        candidate: selected,
        source: AttentionStaticCostModel,
        static_cost: selected_cost,
      }
  }
}
