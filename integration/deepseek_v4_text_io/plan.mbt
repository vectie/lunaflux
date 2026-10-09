///|
/// Translate the model's learned head semantics into model-neutral precision IR.
pub fn learned_text_io_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
  maximum_output_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.LearnedTextIoPrecision raise {
  @precision.LearnedTextIoPrecision::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_width=spec.hidden_size(),
    stream_count=spec.hyper_connection_width(),
    maximum_rows~,
    maximum_output_rows~,
    normalization_epsilon=spec.rms_norm_epsilon(),
    control_epsilon=spec.hyper_connection_epsilon(),
    control_normalization_epsilon=spec.rms_norm_epsilon(),
    maximum_workspace_bytes~,
  )
}

///|
/// Final norm, vocabulary head, learned function matrix, base, scale. No
/// mean fallback or fabricated learned controls are accepted by this adapter.
pub fn bind_text_io_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let hidden = spec.hidden_size()
  let weights = [
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=hidden,
      ),
      payload_name="norm.weight",
    ),
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=Bf16,
        rows=spec.vocabulary_size(),
        columns=hidden,
      ),
      payload_name="head.weight",
    ),
  ]
  weights.append(bind_head_control_weights(checkpoint, spec))
  weights
}

///|
/// Learned head control planes in function/base/scale order. This boundary is
/// also usable by a head-only execution frame without loading the vocabulary.
pub fn bind_head_control_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let hidden = spec.hidden_size()
  let streams = spec.hyper_connection_width()
  [
    checkpoint.bind_weight(
      representation=@format.Representation::new(
        format=F32,
        rows=streams,
        columns=streams * hidden,
      ),
      payload_name="hc_head_fn",
    ),
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=F32,
        rows=1,
        columns=streams,
      ),
      payload_name="hc_head_base",
    ),
    checkpoint.bind_vector(
      representation=@format.Representation::new(format=F32, rows=1, columns=1),
      payload_name="hc_head_scale",
    ),
  ]
}
