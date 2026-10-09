///|
/// Only the model adapter selects the Flash geometry. Index LayerNorm epsilon
/// is explicit: absence from checkpoint config is not permission to infer it
/// from unrelated DeepSeek implementations or the low-rank RMSNorm epsilon.
pub fn flash_sparse_projection_plan(
  spec : @spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_queries~ : Int,
  maximum_history~ : Int,
  index_layer_norm_epsilon~ : Double,
  maximum_attention_workspace_bytes~ : Int64,
  maximum_projection_workspace_bytes~ : Int64,
) -> @precision.IndexedSparseProjectionPrecision raise {
  guard spec.attention_kind(layer_index) is DeepSeekSparseAttention &&
    spec.qk_rope_head_dimension() == 0 &&
    maximum_history <= spec.context_length() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let attention = @precision.IndexedSparseAttentionPrecision::new(
    index_head_count=spec.index_head_count(),
    index_head_dimension=spec.index_head_dimension(),
    query_head_count=spec.attention_head_count(),
    key_value_head_count=spec.kv_head_count(),
    query_key_dimension=spec.qk_nope_head_dimension(),
    value_dimension=spec.value_head_dimension(),
    pool_size=spec.index_kpool(),
    top_k=spec.index_topk(),
    append_incomplete_tail=spec.index_pool_always_select_tail(),
    maximum_queries~,
    maximum_history~,
    maximum_workspace_bytes=maximum_attention_workspace_bytes,
  )
  @precision.IndexedSparseProjectionPrecision::new(
    attention,
    hidden_width=spec.hidden_size(),
    query_rank=spec.q_lora_rank(),
    key_value_rank=spec.kv_lora_rank(),
    rms_epsilon=spec.rms_norm_epsilon(),
    index_layer_norm_epsilon~,
    maximum_workspace_bytes=maximum_projection_workspace_bytes,
  )
}
