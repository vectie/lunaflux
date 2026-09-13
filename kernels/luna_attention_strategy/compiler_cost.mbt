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
fn attention_candidate_online_fold_steps(
  problem : AttentionCompileProblem,
  candidate : AttentionTileCandidate,
) -> Int {
  let (_, context) = problem.dimensions()
  let (_, transfer_rows, _) = candidate.tile.dimensions()
  match candidate.compute {
    AttentionMatrixDot | AttentionMatrixQkPv =>
      attention_ceil_div(context, transfer_rows)
    AttentionGroupedQuerySplitSubgroupDot => {
      let (query_heads, kv_heads, _, _) = problem.shape.dimensions()
      let splits = candidate.subgroups_per_block / (query_heads / kv_heads)
      attention_ceil_div(context, splits)
    }
    AttentionScalarDot
    | AttentionSubgroupDot
    | AttentionGroupedQuerySubgroupDot => context
  }
}

///|
fn attention_candidate_static_cost(
  problem : AttentionCompileProblem,
  candidate : AttentionTileCandidate,
) -> Int64 raise AttentionStrategyError {
  let (query_rows, context_tokens) = problem.dimensions()
  let batch_rows = problem.batch_rows().to_int64()
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
    AttentionGroupedQuerySubgroupDot =>
      candidate.subgroups_per_block.to_int64() * 8L
    AttentionGroupedQuerySplitSubgroupDot =>
      candidate.subgroups_per_block.to_int64() * 8L
    AttentionMatrixDot | AttentionMatrixQkPv =>
      candidate.subgroups_per_block.to_int64() * 64L
  }
  let arithmetic_cost = attention_checked_product(
      dot_work,
      batch_rows,
      Candidate,
    ) /
    arithmetic_divisor
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
    AttentionAsyncStagedShared =>
      // Matrix overlap competes with occupancy because both ring slots stay
      // live. Do not assume a twofold gain without a measured record; preserve
      // the synchronous default while autotune can select actual winners.
      match candidate.compute {
        AttentionMatrixDot | AttentionMatrixQkPv => 2L
        _ => 1L
      }
  }
  let (query_heads, key_value_heads, _, _) = problem.shape.dimensions()
  let query_head_group = match candidate.compute {
    AttentionGroupedQuerySubgroupDot | AttentionGroupedQuerySplitSubgroupDot =>
      (query_heads / key_value_heads).to_int64()
    AttentionScalarDot
    | AttentionSubgroupDot
    | AttentionMatrixDot
    | AttentionMatrixQkPv => 1L
  }
  let memory_cost = attention_checked_product(
      attention_checked_product(key_value_bytes, memory_multiplier, Candidate),
      batch_rows,
      Candidate,
    ) /
    query_head_group
  let scheduling_cost = attention_checked_product(
    attention_checked_product(
      attention_checked_product(query_tiles, batch_rows, Candidate),
      query_heads.to_int64() / query_head_group,
      Candidate,
    ),
    4096L,
    Candidate,
  )
  // Matrix attention folds one vector of scores per tile. Subgroup attention
  // folds individual keys (or a disjoint ordered key subsequence), regardless
  // of how many transfers carry those keys. Retiling staging memory must not
  // invent extra semantic softmax work. Transfer synchronization is separate.
  let fold_steps = attention_candidate_online_fold_steps(problem, candidate).to_int64()
  let fold_cost = attention_checked_product(
    attention_checked_product(
      attention_checked_product(query_tiles, fold_steps, Candidate),
      query_heads.to_int64(),
      Candidate,
    ),
    attention_checked_product(batch_rows, 1024L, Candidate),
    Candidate,
  )
  let transfer_cost = attention_checked_product(
    attention_checked_product(query_tiles, key_value_tiles, Candidate),
    attention_checked_product(batch_rows, 256L, Candidate),
    Candidate,
  )
  attention_checked_add(
    attention_checked_add(
      attention_checked_add(arithmetic_cost, memory_cost, Candidate),
      scheduling_cost,
      Candidate,
    ),
    attention_checked_add(fold_cost, transfer_cost, Candidate),
    Candidate,
  )
}
