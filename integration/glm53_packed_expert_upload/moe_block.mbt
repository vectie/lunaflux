///|
/// Complete recurrent GLM block with automatic routing, packed NVFP4 routed
/// experts, BF16 shared expert and both residual-stream envelopes. All experts
/// for this layer are resident locally; partitioned execution requires the
/// existing collective-aware MoE program and is not silently substituted here.
pub fn GlmRecurrentLayer::with_moe_block(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
) -> Self raise {
  guard spec.feed_forward_kind(layer_index) is RoutedMixtureOfExperts else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let branch = GlmRecurrentLayer::with_hyper_connection(
    spec,
    layer_index~,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    maximum_state_bytes~,
  )
  let routed = expert_compute_plan(
    layer_index~,
    expert_indices=Array::makei(spec.routed_expert_count(), i => i),
    maximum_rows~,
    memory_ceiling=maximum_expert_bytes,
    accuracy~,
  )
  let shared = shared_expert_compute_plan(
    layer_index~,
    maximum_rows~,
    memory_ceiling=maximum_expert_bytes,
    accuracy~,
  )
  guard routed.bank_bytes() <= maximum_expert_bytes &&
    shared.bank_bytes() <= maximum_expert_bytes - routed.bank_bytes() else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  let program = @packed_expert_mlp.MoeProgram::new(
    @expert_source.packed_expert_mlp_source(
      routed,
      symbol="glm_routed_" + layer_index.to_string(),
    ),
    @expert_source.packed_expert_mlp_source(
      shared,
      symbol="glm_shared_" + layer_index.to_string(),
    ),
    @combine_source.moe_combine_source(
      @precision_ir.MoeCombinePrecision::new(
        maximum_rows~,
        hidden=spec.hidden_size(),
        rounding=Bf16RoutedThenSum,
        memory_ceiling=maximum_rows.to_int64() *
          spec.hidden_size().to_int64() *
          10L,
      ),
      symbol="glm_moe_combine_" + layer_index.to_string(),
    ),
    routing=@router_source.routing_source(
      routing_compute_plan(
        maximum_rows~,
        maximum_workspace_bytes=maximum_expert_bytes,
        accuracy~,
      ),
      symbol="glm_router_" + layer_index.to_string(),
    ),
  )
  let envelope = @precision_ir.HyperConnectionPrecision::new(
    stream_count=spec.hyper_connection_streams(),
    hidden_width=spec.hidden_size(),
    maximum_rows~,
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    mix_law=RoundedBf16ControlsAndProducts,
  )
  {
    spec,
    layer_index,
    program: branch.program,
    weights: branch.weights,
    hyper: branch.hyper,
    ffn: Some({
      envelope: @hyper.HyperConnectionFrame::new(
        @hyper_source.hyper_connection_source(
          envelope,
          symbol_prefix="glm_hyper_ffn_" + layer_index.to_string(),
        ),
      ),
      branch: Moe({
        routed,
        shared,
        program,
        source: program.source_bytes(),
        banks: [],
      }),
      input_bytes: maximum_rows.to_int64() *
      spec.hyper_connection_streams().to_int64() *
      spec.hidden_size().to_int64() *
      2L,
      input: None,
      launches: [],
    }),
    status: Fresh,
  }
}
