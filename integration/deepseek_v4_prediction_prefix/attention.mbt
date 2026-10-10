///|
/// Model-specific dimensions select a shared read-set/ownership plan. The
/// caller supplies preceding normalized/rotary/precision-simulated operands.
pub fn prediction_attention_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_main_rows~ : Int,
  maximum_positions~ : Int,
  maximum_state_bytes~ : Int64,
) -> @precision.CommittedDraftKvPrecision raise {
  guard spec.has_dspark() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  @precision.CommittedDraftKvPrecision::new(
    query_heads=spec.attention_head_count(),
    head_width=spec.head_dimension(),
    window_tokens=spec.sliding_window(),
    maximum_main_rows~,
    maximum_draft_rows=spec.dspark_block_size(),
    maximum_positions~,
    maximum_state_bytes~,
  )
}

///|
/// Constructs both phase tables at startup, sharing one persistent ring. AOT
/// linking includes source bytes from both phase renderers using these names.
pub fn prediction_attention_frame(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_main_rows~ : Int,
  maximum_positions~ : Int,
  maximum_state_bytes~ : Int64,
  symbol_prefix~ : String,
) -> @draft_frame.CommittedDraftKvFrame raise {
  let p = prediction_attention_plan(
    spec,
    maximum_main_rows~,
    maximum_positions~,
    maximum_state_bytes~,
  )
  @draft_frame.CommittedDraftKvFrame::new(
    @draft_source.committed_draft_kv_source(
      p,
      phase=Prime,
      symbol_prefix=symbol_prefix + "_prime",
    ),
    @draft_source.committed_draft_kv_source(
      p,
      phase=Predict,
      symbol_prefix=symbol_prefix + "_predict",
    ),
    committed=@draft_source.committed_draft_kv_source(
      p,
      phase=PredictCommitted,
      symbol_prefix=symbol_prefix + "_committed",
    ),
  )
}

///|
/// Prediction blocks have no compressed-history rotary policy. Inverse RoPE
/// and both compact output projections retain the same shared arithmetic plan
/// as base attention, with draft-row geometry and base frequencies.
pub fn prediction_attention_output_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_positions~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.GroupedAttentionOutputPrecision raise {
  require_prediction_stage(spec, 0)
  @precision.GroupedAttentionOutputPrecision::new(
    head_count=spec.attention_head_count(),
    head_width=spec.head_dimension(),
    rotary_width=spec.rope_head_dimension(),
    group_count=spec.output_group_count(),
    output_rank=spec.o_lora_rank(),
    hidden_width=spec.hidden_size(),
    maximum_rows=spec.dspark_block_size(),
    maximum_positions~,
    frequency=Base(theta=spec.rope_theta()),
    maximum_workspace_bytes~,
  )
}
