///|
/// Five full vocabulary rows, with the last prediction stage's learned mHC
/// collapse. The base head's last-row shortcut is not a prediction head.
pub fn prediction_text_head_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_workspace_bytes~ : Int64,
) -> @precision.LearnedTextIoPrecision raise {
  require_prediction_stage(spec, spec.prediction_stage_count() - 1)
  @precision.LearnedTextIoPrecision::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_width=spec.hidden_size(),
    stream_count=spec.hyper_connection_width(),
    maximum_rows=spec.dspark_block_size(),
    maximum_output_rows=spec.dspark_block_size(),
    normalization_epsilon=spec.rms_norm_epsilon(),
    control_epsilon=spec.hyper_connection_epsilon(),
    control_normalization_epsilon=spec.rms_norm_epsilon(),
    maximum_workspace_bytes~,
  )
}

///|
pub fn prediction_markov_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  temperature~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> @precision.SequentialMarkovPrecision raise {
  require_prediction_stage(spec, spec.prediction_stage_count() - 1)
  @precision.SequentialMarkovPrecision::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_width=spec.hidden_size(),
    embedding_width=spec.dspark_markov_rank(),
    maximum_rows=spec.dspark_block_size(),
    temperature~,
    maximum_workspace_bytes~,
  )
}

///|
/// Norm, learned mHC controls, Markov gather/projection and confidence. The
/// global vocabulary matrix is borrowed from base egress, not loaded again.
pub fn bind_prediction_head_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let prefix = @spec.DeepSeekV4BlockAddress::PredictionStage(
    spec.prediction_stage_count() - 1,
  ).checkpoint_prefix(spec)
  let hidden = spec.hidden_size()
  let streams = spec.hyper_connection_width()
  let rank = spec.dspark_markov_rank()
  [
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=hidden,
      ),
      payload_name=prefix + "norm.weight",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=F32,
        rows=streams,
        columns=streams * hidden,
      ),
      payload_name=prefix + "hc_head_fn",
    ),
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=F32,
        rows=1,
        columns=streams,
      ),
      payload_name=prefix + "hc_head_base",
    ),
    checkpoint.bind_vector(
      representation=@format.Representation::new(format=F32, rows=1, columns=1),
      payload_name=prefix + "hc_head_scale",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.vocabulary_size(),
        columns=rank,
      ),
      payload_name=prefix + "markov_head.markov_w1.weight",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.vocabulary_size(),
        columns=rank,
      ),
      payload_name=prefix + "markov_head.markov_w2.weight",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=hidden + rank,
      ),
      payload_name=prefix + "confidence_head.proj.weight",
    ),
  ]
}
