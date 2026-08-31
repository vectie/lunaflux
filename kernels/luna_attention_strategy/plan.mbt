///|
pub fn AttentionShape::new(
  query_heads~ : Int,
  kv_heads~ : Int,
  head_dimension~ : Int,
  tokens_per_page~ : Int,
) -> AttentionShape raise AttentionStrategyError {
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    query_heads % kv_heads != 0 ||
    head_dimension <= 0 ||
    tokens_per_page <= 0 {
    raise InvalidAttentionStrategy(Shape)
  }
  { query_heads, kv_heads, head_dimension, tokens_per_page }
}

///|
pub fn plan_attention_dispatch(
  shape : AttentionShape,
  maximum_context_tokens : Int,
  maximum_prefill_tokens : Int,
  maximum_key_value_splits : Int,
) -> AttentionDispatchPlan raise AttentionStrategyError {
  if maximum_context_tokens <= 0 ||
    maximum_prefill_tokens <= 0 ||
    maximum_key_value_splits <= 0 ||
    maximum_key_value_splits > 64 {
    raise InvalidAttentionStrategy(Shape)
  }
  let query_tile = if maximum_prefill_tokens >= 16 {
    16
  } else if maximum_prefill_tokens >= 8 {
    8
  } else {
    1
  }
  let desired_splits = if maximum_context_tokens >= 8192 {
    8
  } else if maximum_context_tokens >= 4096 {
    4
  } else if maximum_context_tokens >= 2048 {
    2
  } else {
    1
  }
  let allowed_splits = if desired_splits < maximum_key_value_splits {
    desired_splits
  } else {
    maximum_key_value_splits
  }
  let maximum_pages = 1 + (maximum_context_tokens - 1) / shape.tokens_per_page
  let splits = if allowed_splits < maximum_pages {
    allowed_splits
  } else {
    maximum_pages
  }
  let prefill = {
    phase: Prefill,
    traversal: QueryTile,
    reduction: OnlineSoftmax,
    query_tile,
    key_value_splits: 1,
  }
  let decode = {
    phase: Decode,
    traversal: KeyValueSplit,
    reduction: if splits == 1 {
      OnlineSoftmax
    } else {
      SplitOnlineSoftmaxMerge
    },
    query_tile: 1,
    key_value_splits: splits,
  }
  { shape, prefill, decode }
}

///|
fn attention_checked_product(
  left : Int64,
  right : Int64,
  rule : AttentionStrategyRule,
) -> Int64 raise AttentionStrategyError {
  if left < 0L ||
    right < 0L ||
    (right != 0L && left > 9223372036854775807L / right) {
    raise InvalidAttentionStrategy(rule)
  }
  left * right
}

///|
/// Plans the bounded partial-online-softmax workspace for decode split-K.
///
/// Partition boundaries are page aligned and merge order is exactly ascending
/// `split_index`. A one-split plan covers `[0, maximum_context_tokens)` and is
/// numerically the ordinary online-softmax reduction expressed as one partial.
pub fn plan_decode_attention_workspace(
  shape : AttentionShape,
  decode : AttentionPhasePlan,
  maximum_rows : Int,
  maximum_context_tokens : Int,
  maximum_workspace_bytes : Int64,
) -> AttentionDecodeWorkspacePlan raise AttentionStrategyError {
  if decode.phase != Decode ||
    decode.traversal != KeyValueSplit ||
    maximum_rows <= 0 ||
    maximum_context_tokens <= 0 ||
    maximum_workspace_bytes <= 0L {
    raise InvalidAttentionStrategy(Workspace)
  }
  let split_count = decode.key_value_splits
  if split_count <= 0 || split_count > 64 {
    raise InvalidAttentionStrategy(Split)
  }
  if (split_count == 1 && decode.reduction != OnlineSoftmax) ||
    (split_count > 1 && decode.reduction != SplitOnlineSoftmaxMerge) {
    raise InvalidAttentionStrategy(Workspace)
  }
  let page_count = 1 + (maximum_context_tokens - 1) / shape.tokens_per_page
  if page_count <= 0 || split_count > page_count {
    raise InvalidAttentionStrategy(Partition)
  }
  let base_pages_per_split = page_count / split_count
  let larger_split_count = page_count % split_count
  let partitions : Array[AttentionKeyValuePartition] = []
  for split_index in 0..<split_count {
    let larger_before = if split_index < larger_split_count {
      split_index
    } else {
      larger_split_count
    }
    let begin_page = split_index * base_pages_per_split + larger_before
    let partition_pages = base_pages_per_split +
      (if split_index < larger_split_count { 1 } else { 0 })
    let end_page = begin_page + partition_pages
    let begin_token = begin_page * shape.tokens_per_page
    let page_end_token = end_page * shape.tokens_per_page
    let end_token = if page_end_token < maximum_context_tokens {
      page_end_token
    } else {
      maximum_context_tokens
    }
    if begin_token >= end_token ||
      (split_index > 0 && begin_token % shape.tokens_per_page != 0) {
      raise InvalidAttentionStrategy(Partition)
    }
    partitions.push({ split_index, begin_token, end_token })
  }
  let partial_count = attention_checked_product(
    attention_checked_product(
      maximum_rows.to_int64(),
      shape.query_heads.to_int64(),
      Workspace,
    ),
    split_count.to_int64(),
    Workspace,
  )
  let scalar_region_bytes = attention_checked_product(
    partial_count,
    4L,
    Workspace,
  )
  let numerator_bytes = attention_checked_product(
    attention_checked_product(
      partial_count,
      shape.head_dimension.to_int64(),
      Workspace,
    ),
    4L,
    Workspace,
  )
  if scalar_region_bytes > 9223372036854775807L - scalar_region_bytes ||
    scalar_region_bytes * 2L > 9223372036854775807L - numerator_bytes {
    raise InvalidAttentionStrategy(Workspace)
  }
  let total_bytes = scalar_region_bytes * 2L + numerator_bytes
  if total_bytes > maximum_workspace_bytes {
    raise InvalidAttentionStrategy(Workspace)
  }
  {
    maximum_rows,
    maximum_context_tokens,
    split_count,
    partial_count,
    maximum_offset_bytes: 0L,
    denominator_offset_bytes: scalar_region_bytes,
    numerator_offset_bytes: scalar_region_bytes * 2L,
    total_bytes,
    partitions: ReadOnlyArray::from_array(partitions),
  }
}
