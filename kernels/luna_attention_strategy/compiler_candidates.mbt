///|
fn attention_candidate_shared_bytes(
  problem : AttentionCompileProblem,
  compute : AttentionTileCompute,
  query_tile : Int,
  key_value_tile : Int,
  pipeline_stages : Int,
  memory : AttentionTileMemory,
  subgroups : Int,
) -> Int64 raise AttentionStrategyError {
  if memory == AttentionDirectGlobal {
    return 0L
  }
  let (_, _, head_dimension, _) = problem.shape.dimensions()
  if compute == AttentionGroupedQueryMatrixQkPv {
    return (16L + 2L * key_value_tile.to_int64() * pipeline_stages.to_int64()) *
      head_dimension.to_int64() *
      2L +
      16L
  }
  if compute == AttentionGroupedQueryBlockwiseDot ||
    compute == AttentionGroupedQueryDualScoreBlockwiseDot ||
    compute == AttentionGroupedQueryFmaBlockwiseDot ||
    compute is AttentionGroupedQueryOwnedBlockwiseDot(..) {
    let (heads, kv_heads, _, _) = problem.shape.dimensions()
    let operand_bytes = attention_checked_product(
      attention_checked_product(
        key_value_tile.to_int64(),
        head_dimension.to_int64(),
        Candidate,
      ),
      4L * pipeline_stages.to_int64(),
      Candidate,
    )
    let scores = attention_checked_product(
      attention_checked_product(
        (heads / kv_heads).to_int64(),
        key_value_tile.to_int64(),
        Candidate,
      ),
      4L,
      Candidate,
    )
    return attention_checked_add(
      attention_checked_add(operand_bytes, scores, Candidate),
      16L,
      Candidate,
    )
  }
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
  if attention_fragment_ownership(
      compute,
      query_rows=query_tile,
      key_rows=key_value_tile,
      head_columns=head_dimension,
      subgroups~,
    ) ==
    QueryOwnedFragments {
    return attention_checked_add(operand_bytes, 16L, Candidate)
  }
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
  let probability_bytes = match compute {
    AttentionMatrixQkPv | AttentionMatrixQkPvApproxExp => 2L
    AttentionScalarDot
    | AttentionSubgroupDot
    | AttentionGroupedQuerySubgroupDot
    | AttentionGroupedQuerySplitSubgroupDot
    | AttentionGroupedQueryBlockwiseDot
    | AttentionGroupedQueryDualScoreBlockwiseDot
    | AttentionGroupedQueryFmaBlockwiseDot
    | AttentionGroupedQueryOwnedBlockwiseDot(..)
    | AttentionGroupedQueryMatrixQkPv
    | AttentionMatrixDot => 4L
  }
  let score_and_weight_bytes = attention_checked_product(
    score_elements,
    4L + probability_bytes,
    Candidate,
  )
  let online_state_bytes = attention_checked_add(
    attention_checked_product(query_tile.to_int64(), 12L, Candidate),
    16L,
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
  let ownership = attention_fragment_ownership(
    candidate.compute,
    query_rows~,
    key_rows=key_value_rows,
    head_columns~,
    subgroups~,
  )
  if candidate.compute == AttentionMatrixQkPvApproxExp &&
    (
      !problem.allow_approximate_exponential ||
      problem.phase != Prefill ||
      ownership != QueryOwnedFragments
    ) {
    return false
  }
  if candidate.retained_query_dimensions < 0 ||
    (
      candidate.joint_query_consumers &&
      candidate.retained_query_dimensions == 0
    ) ||
    candidate.retained_query_dimensions > head_columns ||
    (
      candidate.retained_query_dimensions > 0 &&
      ownership != QueryOwnedFragments
    ) {
    return false
  }
  if problem.required_fragment_ownership is Some(required) &&
    required != ownership {
    return false
  }
  if ownership == QueryOwnedFragments &&
    (
      !problem.allow_alternative_softmax_reduction ||
      problem.numeric_type != AttentionBFloat16
    ) {
    return false
  }
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
    AttentionGroupedQuerySubgroupDot
    | AttentionGroupedQuerySplitSubgroupDot
    | AttentionGroupedQueryBlockwiseDot
    | AttentionGroupedQueryDualScoreBlockwiseDot
    | AttentionGroupedQueryFmaBlockwiseDot => {
      // The fixed historical laws retain their exact ownership geometry.
      let (query_heads, key_value_heads, _, _) = problem.shape.dimensions()
      let group_size = if key_value_heads > 0 {
        query_heads / key_value_heads
      } else {
        0
      }
      problem.phase == Decode &&
      key_value_heads > 0 &&
      query_heads % key_value_heads == 0 &&
      (match candidate.compute {
        AttentionGroupedQuerySubgroupDot =>
          candidate.subgroups_per_block == group_size
        AttentionGroupedQueryBlockwiseDot
        | AttentionGroupedQueryDualScoreBlockwiseDot
        | AttentionGroupedQueryFmaBlockwiseDot =>
          candidate.subgroups_per_block == group_size &&
          problem.allow_alternative_softmax_reduction &&
          (
            candidate.compute != AttentionGroupedQueryDualScoreBlockwiseDot ||
            capabilities.subgroup_width % 2 == 0
          )
        AttentionGroupedQuerySplitSubgroupDot =>
          candidate.subgroups_per_block > group_size &&
          candidate.subgroups_per_block % group_size == 0
        _ => false
      })
    }
    AttentionGroupedQueryOwnedBlockwiseDot(score_owners~, contraction=_) => {
      let (heads, kv_heads, _, _) = problem.shape.dimensions()
      problem.phase == Decode &&
      kv_heads > 0 &&
      heads % kv_heads == 0 &&
      candidate.subgroups_per_block == heads / kv_heads &&
      problem.allow_alternative_softmax_reduction &&
      (score_owners == 2 || score_owners == 4 || score_owners == 8) &&
      capabilities.subgroup_width % score_owners == 0
    }
    AttentionGroupedQueryMatrixQkPv => {
      let (heads, kv_heads, _, _) = problem.shape.dimensions()
      problem.phase == Decode &&
      problem.numeric_type == AttentionBFloat16 &&
      problem.allow_alternative_softmax_reduction &&
      kv_heads > 0 &&
      heads % kv_heads == 0 &&
      heads / kv_heads <= 16 &&
      candidate.subgroups_per_block == 1 &&
      capabilities.matrix_shape is Some(matrix) &&
      matrix.dimensions() == (16, 16, 16) &&
      (head_dimension == 64 || head_dimension == 128)
    }
    AttentionMatrixDot | AttentionMatrixQkPv | AttentionMatrixQkPvApproxExp =>
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
    tile: { query_rows, key_value_rows, head_columns, },
    subgroups_per_block,
    pipeline_stages,
    vector_bytes,
    retained_query_dimensions: 0,
    joint_query_consumers: false,
    shared_memory_bytes: attention_candidate_shared_bytes(
      problem, compute, query_rows, key_value_rows, pipeline_stages, memory, subgroups_per_block,
    ),
  }
  if attention_candidate_compatible(problem, capabilities, candidate) {
    Some(candidate)
  } else {
    None
  }
}

///|
/// Generates a deterministic, bounded candidate set. This is compiler/startup
/// work; request-time execution consumes only the selected plan.
pub fn generate_attention_tile_candidates(
  problem : AttentionCompileProblem,
  capabilities : AttentionCompilerCapabilities,
) -> ReadOnlyArray[AttentionTileCandidate] raise AttentionStrategyError {
  let (query_rows, context_tokens) = problem.dimensions()
  let specs = attention_candidate_specs()
  specs.append(block_owned_matrix_specs())
  specs.append(query_owned_attention_specs())
  if problem.allow_approximate_exponential {
    specs.append(approximate_query_owned_attention_specs())
  }
  let candidates = specs.filter_map(spec => {
    if spec.phase != problem.phase ||
      // Minimum rows are a profitability heuristic, not a tail-safety limit.
      // An explicit ownership contract covers short masked tails as well.
      (
        query_rows < spec.minimum_query_rows &&
        problem.required_fragment_ownership is None
      ) ||
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
  candidates.append(retained_query_attention_candidates(candidates))
  if problem.phase == Decode {
    let (query_heads, key_value_heads, _, _) = problem.shape.dimensions()
    if key_value_heads > 0 && query_heads % key_value_heads == 0 {
      let query_heads_per_group = query_heads / key_value_heads
      if problem.allow_alternative_softmax_reduction {
        for
          memory_index, memory in [
            AttentionStagedShared,
            AttentionAsyncStagedShared,
          ] {
          for key_index, keys in [32, 64] {
            match
              make_attention_candidate(
                problem,
                capabilities,
                480 + memory_index * 2 + key_index,
                AttentionGroupedQueryMatrixQkPv,
                memory,
                1,
                keys,
                1,
                if memory == AttentionAsyncStagedShared {
                  2
                } else {
                  1
                },
                16,
              ) {
              Some(candidate) => candidates.push(candidate)
              None => ()
            }
          }
        }
        for owner_index, owners in [2, 4, 8] {
          for contract_index, contraction in [false, true] {
            // ID 458 is deliberately vacant: all new laws have fresh IDs.
            // One complete portable family, not a model or device special case.
            for key_index, keys in [32, 64] {
              match
                make_attention_candidate(
                  problem,
                  capabilities,
                  460 + owner_index * 4 + contract_index * 2 + key_index,
                  AttentionGroupedQueryOwnedBlockwiseDot(
                    score_owners=owners,
                    contraction~,
                  ),
                  AttentionAsyncStagedShared,
                  1,
                  keys,
                  query_heads_per_group,
                  2,
                  8,
                ) {
                Some(candidate) => candidates.push(candidate)
                None => ()
              }
            }
          }
        }
        for index, keys in [32, 64] {
          match
            make_attention_candidate(
              problem,
              capabilities,
              456 + index,
              AttentionGroupedQueryFmaBlockwiseDot,
              AttentionAsyncStagedShared,
              1,
              keys,
              query_heads_per_group,
              2,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
        }
        for index, keys in [32, 64] {
          match
            make_attention_candidate(
              problem,
              capabilities,
              454 + index,
              AttentionGroupedQueryDualScoreBlockwiseDot,
              AttentionAsyncStagedShared,
              1,
              keys,
              query_heads_per_group,
              2,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
        }
        for index, keys in [32, 64] {
          match
            make_attention_candidate(
              problem,
              capabilities,
              452 + index,
              AttentionGroupedQueryBlockwiseDot,
              AttentionAsyncStagedShared,
              1,
              keys,
              query_heads_per_group,
              2,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
        }
        for index, keys in [32, 64] {
          match
            make_attention_candidate(
              problem,
              capabilities,
              450 + index,
              AttentionGroupedQueryBlockwiseDot,
              AttentionStagedShared,
              1,
              keys,
              query_heads_per_group,
              1,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
        }
      }
      if query_heads_per_group > 1 {
        match
          make_attention_candidate(
            problem,
            capabilities,
            420,
            AttentionGroupedQuerySubgroupDot,
            AttentionStagedShared,
            1,
            64,
            query_heads_per_group,
            1,
            8,
          ) {
          Some(candidate) => candidates.push(candidate)
          None => ()
        }
        let maximum_subgroups = capabilities.max_threads_per_block /
          capabilities.subgroup_width
        let split_subgroups = Int::min(8, maximum_subgroups)
        if split_subgroups > query_heads_per_group &&
          split_subgroups % query_heads_per_group == 0 {
          match
            make_attention_candidate(
              problem,
              capabilities,
              430,
              AttentionGroupedQuerySplitSubgroupDot,
              AttentionStagedShared,
              1,
              64,
              split_subgroups,
              1,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
          match
            make_attention_candidate(
              problem,
              capabilities,
              440,
              AttentionGroupedQuerySplitSubgroupDot,
              AttentionAsyncStagedShared,
              1,
              64,
              split_subgroups,
              2,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
          match
            make_attention_candidate(
              problem,
              capabilities,
              441,
              AttentionGroupedQuerySplitSubgroupDot,
              AttentionAsyncStagedShared,
              1,
              32,
              split_subgroups,
              2,
              8,
            ) {
            Some(candidate) => candidates.push(candidate)
            None => ()
          }
        }
      }
    }
  }
  if candidates.length() == 0 {
    raise InvalidAttentionStrategy(Capability)
  }
  ReadOnlyArray::from_array(candidates)
}
