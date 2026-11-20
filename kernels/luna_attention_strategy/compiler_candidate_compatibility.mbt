///|
fn attention_candidate_shared_bytes(
  problem : AttentionCompileProblem,
  compute : AttentionTileCompute,
  query_tile : Int32,
  key_value_tile : Int32,
  pipeline_stages : Int32,
  memory : AttentionTileMemory,
  subgroups : Int32,
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
    candidate.retained_query_dimensions > head_columns ||
    (
      (
        candidate.retained_query_dimensions > 0 ||
        candidate.joint_query_consumers
      ) &&
      ownership != QueryOwnedFragments
    ) {
    return false
  }
  if candidate.right_operand_read_slots < 0 ||
    candidate.right_operand_read_slots > 2 ||
    (
      candidate.right_operand_read_slots > 0 &&
      (
        ownership != QueryOwnedFragments ||
        candidate.joint_query_consumers ||
        candidate.retained_query_dimensions > 0 ||
        candidate.right_operand_read_slots > key_value_rows / 16
      )
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
