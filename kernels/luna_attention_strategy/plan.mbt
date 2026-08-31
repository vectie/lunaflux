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
  let splits = if desired_splits < maximum_key_value_splits {
    desired_splits
  } else {
    maximum_key_value_splits
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
