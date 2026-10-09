///|
/// Single-rank low-rank query/shared-KV prefix. Headwise BF16 expression
/// normalization is distinct from the learned F32 RMSNorm boundaries.
pub fn query_kv_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.QuantizedQueryKvPrecision raise {
  @precision.QuantizedQueryKvPrecision::new(
    hidden_width=spec.hidden_size(),
    query_rank=spec.q_lora_rank(),
    head_count=spec.attention_head_count(),
    head_width=spec.head_dimension(),
    maximum_rows~,
    normalization_epsilon=spec.rms_norm_epsilon(),
    activation_amax_floor=0.0001,
    maximum_workspace_bytes~,
  )
}

///|
/// Retains packed E4M3/UE8M0 matrices; norms use their actual BF16 vectors.
/// The returned order is Query-A, query norm, Query-B, KV, KV norm.
pub fn bind_query_kv_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer : Int,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let plan = query_kv_plan(
    spec,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  let matrices = plan.weight_representations()
  let prefix = "layers.\{layer}.attn."
  let bind = (name, representation) => {
    checkpoint.bind_weight(
      representation~,
      payload_name=prefix + name + ".weight",
      scales_name=prefix + name + ".scale",
    )
  }
  [
    bind("wq_a", matrices[0]),
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=spec.q_lora_rank(),
      ),
      payload_name=prefix + "q_norm.weight",
    ),
    bind("wq_b", matrices[1]),
    bind("wkv", matrices[2]),
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=spec.head_dimension(),
      ),
      payload_name=prefix + "kv_norm.weight",
    ),
  ]
}
