///|
/// Full attention sublayer: four residual streams → mHC collapse → weighted
/// input norm → KDA branch → mHC residual publication in one prepared queue.
pub fn GlmRecurrentLayer::with_hyper_connection(
  spec : @glm53_spec.Glm53FlashSpec,
  layer_index~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
) -> Self raise {
  let branch = GlmRecurrentLayer::new(
    spec,
    layer_index~,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    maximum_state_bytes~,
  )
  let plan = @precision_ir.HyperConnectionPrecision::new(
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
    hyper: Some(
      @hyper.HyperConnectionFrame::new(
        @hyper_source.hyper_connection_source(
          plan,
          symbol_prefix="glm_hyper_attn_" + layer_index.to_string(),
        ),
      ),
    ),
    ffn: None,
    status: Fresh,
  }
}

///|
fn bind_attention_hyper_weights(
  checkpoint : @precision_materialize.PackedCheckpoint,
  spec : @glm53_spec.Glm53FlashSpec,
  layer : Int,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let prefix = "model.language_model.layers.\{layer}."
  let streams = spec.hyper_connection_streams()
  let mix = streams * (streams + 2)
  let weights = [
    checkpoint.bind_weight(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows=mix,
        columns=streams * spec.hidden_size(),
      ),
      payload_name=prefix + "hc_attn_fn",
    ),
  ]
  for (name, width) in [("hc_attn_base", mix), ("hc_attn_scale", 3)] {
    weights.push(
      checkpoint.bind_vector(
        representation=@precision_format.Representation::new(
          format=F32,
          rows=1,
          columns=width,
        ),
        payload_name=prefix + name,
      ),
    )
  }
  weights.push(
    checkpoint.bind_vector(
      representation=@precision_format.Representation::new(
        format=Bf16,
        rows=1,
        columns=spec.hidden_size(),
      ),
      payload_name=prefix + "input_layernorm.weight",
    ),
  )
  weights
}
