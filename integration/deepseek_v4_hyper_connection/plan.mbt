///|
pub(all) enum DeepSeekHyperBranch {
  Attention
  FeedForward
}

///|
pub fn hyper_connection_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
  control_square_law? : @precision.SquareReductionLaw = OrderedFloat32,
  control_dot_law? : @precision.DotReductionLaw = OrderedFloat32,
  control_vector_affine_law? : @precision.Float32AffineLaw = SeparateMultiplyAdd,
  control_sigmoid_law? : @precision.SigmoidEvaluationLaw = StableBranches,
  control_matrix_affine_law? : @precision.Float32AffineLaw = SeparateMultiplyAdd,
  matrix_sum_law? : @precision.SumReductionLaw = OrderedFloat32,
  control_root_law? : @precision.ReciprocalRootLaw = DivideRoundedSqrtFloat32,
  normalization_square_law? : @precision.SquareReductionLaw = VectorizedTreeFloat32(
    partitions=256,
    vector_width=1,
  ),
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
    control_square_law~,
    control_dot_law~,
    control_vector_affine_law~,
    control_sigmoid_law~,
    control_matrix_affine_law~,
    matrix_sum_law~,
    control_root_law~,
    normalization_square_law~,
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
