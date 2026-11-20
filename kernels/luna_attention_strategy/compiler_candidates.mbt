///|
fn make_attention_candidate(
  problem : AttentionCompileProblem,
  capabilities : AttentionCompilerCapabilities,
  stable_id : Int32,
  compute : AttentionTileCompute,
  memory : AttentionTileMemory,
  query_rows : Int32,
  key_value_rows : Int32,
  subgroups_per_block : Int32,
  pipeline_stages : Int32,
  vector_bytes : Int32,
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
    right_operand_read_slots: 0,
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
  let read_variants = contiguous_right_read_candidates(candidates)
  candidates.append(retained_query_attention_candidates(candidates))
  candidates.append(read_variants)
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
        let split_subgroups = Int32::min(8, maximum_subgroups)
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
