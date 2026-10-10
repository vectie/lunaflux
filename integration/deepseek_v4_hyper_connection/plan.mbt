///|
pub(all) enum DeepSeekHyperBranch {
  Attention
  FeedForward
}

///|
pub fn hyper_connection_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
) -> @precision.HyperConnectionPrecision raise {
  @precision.HyperConnectionPrecision::new(
    stream_count=spec.hyper_connection_width(),
    hidden_width=spec.hidden_size(),
    maximum_rows~,
    sinkhorn_iterations=spec.hyper_connection_sinkhorn_iterations(),
    control_epsilon=spec.hyper_connection_epsilon(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    control_law=F32ProjectionThenNormalize,
    mix_law=TransposedF32ResidualThenBranch,
  )
}

///|
pub fn bind_hyper_connection_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  branch~ : DeepSeekHyperBranch,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  bind_block_hyper_connection_weights(
    checkpoint,
    spec,
    block=BaseLayer(layer),
    branch~,
  )
}

///|
pub fn bind_block_hyper_connection_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
  block~ : @spec.DeepSeekV4BlockAddress,
  branch~ : DeepSeekHyperBranch,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let streams = spec.hyper_connection_width()
  let mix = streams * (streams + 2)
  let name = match branch {
    Attention => "attn"
    FeedForward => "ffn"
  }
  let prefix = block.checkpoint_prefix(spec)
  let weights = [
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=F32,
        rows=mix,
        columns=streams * spec.hidden_size(),
      ),
      payload_name=prefix + "hc_" + name + "_fn",
    ),
  ]
  for (suffix, width) in [("base", mix), ("scale", 3)] {
    weights.push(
      checkpoint.bind_vector(
        representation=@format.Representation::new(
          format=F32,
          rows=1,
          columns=width,
        ),
        payload_name=prefix + "hc_" + name + "_" + suffix,
      ),
    )
  }
  weights.push(
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=spec.hidden_size(),
      ),
      payload_name=prefix + name + "_norm.weight",
    ),
  )
  weights
}
