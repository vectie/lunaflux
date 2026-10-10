///|
fn require_prediction_stage(
  spec : @spec.DeepSeekV4ModelSpec,
  stage : Int,
) -> Unit raise {
  guard spec.has_dspark() && stage >= 0 && stage < spec.prediction_stage_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
}

///|
/// Query-A, learned query norm, Query-B, shared KV, learned KV norm, sink,
/// grouped Output-A and Output-B.
/// Main and draft consumers share the one KV matrix/vector allocation.
pub fn bind_prediction_attention_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  stage~ : Int,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  require_prediction_stage(spec, stage)
  let q = @precision.QuantizedQueryKvPrecision::new(
    hidden_width=spec.hidden_size(),
    query_rank=spec.q_lora_rank(),
    head_count=spec.attention_head_count(),
    head_width=spec.head_dimension(),
    maximum_rows=1,
    normalization_epsilon=spec.rms_norm_epsilon(),
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=1048576L,
  )
  let matrices = q.weight_representations()
  let output = prediction_attention_output_plan(
    spec,
    maximum_positions=spec.max_position_embeddings(),
    maximum_workspace_bytes=1048576L,
  ).weight_representations()
  let prefix = "mtp.\{stage}.attn."
  let matrix = (name, rep) => {
    checkpoint.bind_weight(
      representation=rep,
      payload_name=prefix + name + ".weight",
      scales_name=prefix + name + ".scale",
    )
  }
  let vector = (name, columns, format) => {
    checkpoint.bind_vector(
      representation=@format.Representation::new(format~, rows=1, columns~),
      payload_name=prefix + name,
    )
  }
  [
    matrix("wq_a", matrices[0]),
    vector("q_norm.weight", spec.q_lora_rank(), Bf16),
    matrix("wq_b", matrices[1]),
    matrix("wkv", matrices[2]),
    vector("kv_norm.weight", spec.head_dimension(), Bf16),
    vector("attn_sink", spec.attention_head_count(), F32),
    matrix("wo_a", output[0]),
    matrix("wo_b", output[1]),
  ]
}
