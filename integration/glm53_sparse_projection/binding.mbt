///|
/// Exact twelve BF16 checkpoint operands in generic projection-plan order.
/// No fabricated biases/scales; one-dimensional normalization tensors keep
/// their stored rank. Pool APE and output projection are independent operands.
pub fn bind_flash_sparse_projection_weights(
  checkpoint : @checkpoint.PackedCheckpoint,
  spec : @spec.Glm53FlashSpec,
  layer_index~ : Int,
) -> Array[@checkpoint.PackedCheckpointWeight] raise {
  guard spec.attention_kind(layer_index) is DeepSeekSparseAttention &&
    spec.qk_rope_head_dimension() == 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let prefix = "model.language_model.layers.\{layer_index}.self_attn."
  let hidden = spec.hidden_size()
  let qr = spec.q_lora_rank()
  let kr = spec.kv_lora_rank()
  let index = spec.index_head_dimension()
  let parameters = [
    ("q_a_proj.weight", qr, hidden, false),
    ("q_a_layernorm.weight", 1, qr, true),
    (
      "q_b_proj.weight",
      spec.attention_head_count() * spec.qk_nope_head_dimension(),
      qr,
      false,
    ),
    ("kv_a_proj_with_mqa.weight", kr, hidden, false),
    ("kv_a_layernorm.weight", 1, kr, true),
    (
      "kv_b_proj.weight",
      spec.kv_head_count() *
      (spec.qk_nope_head_dimension() + spec.value_head_dimension()),
      kr,
      false,
    ),
    ("indexer.wq_b.weight", spec.index_head_count() * index, qr, false),
    ("indexer.wk.weight", index, hidden, false),
    ("indexer.k_norm.weight", 1, index, true),
    ("indexer.k_norm.bias", 1, index, true),
    ("indexer.weights_proj.weight", spec.index_head_count(), hidden, false),
    ("indexer.index_kpool_compress_gate", index, hidden, false),
  ]
  parameters.map(parameter => {
    let (name, rows, columns, vector) = parameter
    let representation = @format.Representation::new(
      format=Bf16,
      rows~,
      columns~,
    )
    if vector {
      checkpoint.bind_vector(representation~, payload_name=prefix + name)
    } else {
      checkpoint.bind_weight(representation~, payload_name=prefix + name)
    }
  })
}

///|
/// Output projection and learned pool APE are retained in their stored shape.
/// Returned order is output weight then pool APE, following the twelve input
/// projection operands in the complete block's binding list.
pub fn bind_flash_sparse_output_weights(
  checkpoint : @checkpoint.PackedCheckpoint,
  spec : @spec.Glm53FlashSpec,
  layer_index~ : Int,
) -> Array[@checkpoint.PackedCheckpointWeight] raise {
  guard spec.attention_kind(layer_index) is DeepSeekSparseAttention &&
    spec.qk_rope_head_dimension() == 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let prefix = "model.language_model.layers.\{layer_index}.self_attn."
  [
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.hidden_size(),
        columns=spec.attention_head_count() * spec.value_head_dimension(),
      ),
      payload_name=prefix + "o_proj.weight",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.index_kpool(),
        columns=spec.index_head_dimension(),
      ),
      payload_name=prefix + "indexer.index_kpool_compress_ape",
    ),
  ]
}
