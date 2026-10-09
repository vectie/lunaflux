///|
fn all_finite(values : ArrayView[Double]) -> Bool {
  values.all(value => !value.is_nan() && !value.is_inf())
}

///|
pub fn JointTinyTransformerLimits::new(
  maximum_rows~ : Int,
  maximum_hidden_width~ : Int,
  maximum_feed_forward_width~ : Int,
  maximum_elements~ : Int,
) -> Self raise JointTinyTransformerReferenceError {
  if maximum_rows <= 0 ||
    maximum_rows > 128 ||
    maximum_hidden_width <= 0 ||
    maximum_hidden_width > 64 ||
    maximum_feed_forward_width <= 0 ||
    maximum_feed_forward_width > 256 ||
    maximum_elements <= 0 ||
    maximum_elements > 1048576 {
    raise InvalidLimit
  }
  {
    maximum_rows,
    maximum_hidden_width,
    maximum_feed_forward_width,
    maximum_elements,
  }
}

///|
pub fn JointTinyConditioningMatrix::new(
  source~ : JointTinyConditioningSource,
  rows~ : Int,
  width~ : Int,
  values~ : ArrayView[Double],
) -> Self raise JointTinyTransformerReferenceError {
  if rows < 0 || rows > 64 || width <= 0 || width > 64 {
    raise InvalidConditioningShape(source)
  }
  if rows.to_int64() * width.to_int64() != values.length().to_int64() {
    raise InvalidConditioningShape(source)
  }
  if !all_finite(values) {
    raise NonFiniteInput
  }
  { source, rows, width, values: ReadOnlyArray::from_array(values), }
}

///|
pub fn JointTinyConditioningInput::new(
  text~ : JointTinyConditioningMatrix,
  keyframe~ : JointTinyConditioningMatrix,
  reference~ : JointTinyConditioningMatrix,
) -> Self raise JointTinyTransformerReferenceError {
  if text.source != TextConditioning ||
    text.rows <= 0 ||
    keyframe.source != KeyframeConditioning ||
    reference.source != ReferenceConditioning {
    raise InvalidConditioningContract
  }
  { text, keyframe, reference, }
}

///|
pub fn JointTinyDense::new(
  input_width~ : Int,
  output_width~ : Int,
  weights~ : ArrayView[Double],
  bias~ : ArrayView[Double],
) -> Self raise JointTinyTransformerReferenceError {
  if input_width <= 0 ||
    input_width > 256 ||
    output_width <= 0 ||
    output_width > 256 ||
    input_width.to_int64() * output_width.to_int64() !=
    weights.length().to_int64() ||
    bias.length() != output_width {
    raise InvalidDenseShape
  }
  if !all_finite(weights) || !all_finite(bias) {
    raise NonFiniteInput
  }
  {
    input_width,
    output_width,
    weights: ReadOnlyArray::from_array(weights),
    bias: ReadOnlyArray::from_array(bias),
  }
}

///|
pub fn JointTinyMlp::new(
  up~ : JointTinyDense,
  down~ : JointTinyDense,
) -> Self raise JointTinyTransformerReferenceError {
  if up.output_width != down.input_width || up.input_width != down.output_width {
    raise InvalidMlpShape
  }
  { up, down, }
}

///|
fn dense_shape(dense : JointTinyDense, input : Int, output : Int) -> Bool {
  dense.input_width == input && dense.output_width == output
}

///|
pub fn JointTinyTransformerWeights::new(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  keyframe_input_width~ : Int,
  reference_input_width~ : Int,
  text_projection~ : JointTinyDense,
  keyframe_projection~ : JointTinyDense,
  reference_projection~ : JointTinyDense,
  video_input_projection~ : JointTinyDense,
  audio_input_projection~ : JointTinyDense,
  query_projection~ : JointTinyDense,
  key_projection~ : JointTinyDense,
  value_projection~ : JointTinyDense,
  attention_output_projection~ : JointTinyDense,
  video_mlp~ : JointTinyMlp,
  audio_mlp~ : JointTinyMlp,
  video_output_projection~ : JointTinyDense,
  audio_output_projection~ : JointTinyDense,
) -> Self raise JointTinyTransformerReferenceError {
  let denoiser = plan.denoiser()
  let shapes = plan.shapes()
  let hidden = denoiser.hidden_size()
  let video_width = shapes.video_latent_channels() *
    shapes.video_patch_temporal() *
    shapes.video_patch_height() *
    shapes.video_patch_width()
  let audio_width = shapes.audio_latent_channels()
  if plan.schema_version() != 1 ||
    keyframe_input_width <= 0 ||
    reference_input_width <= 0 ||
    !dense_shape(
      text_projection,
      denoiser.text_conditioning_dimension(),
      hidden,
    ) ||
    !dense_shape(keyframe_projection, keyframe_input_width, hidden) ||
    !dense_shape(reference_projection, reference_input_width, hidden) ||
    !dense_shape(video_input_projection, video_width, hidden) ||
    !dense_shape(audio_input_projection, audio_width, hidden) ||
    !dense_shape(query_projection, hidden, hidden) ||
    !dense_shape(key_projection, hidden, hidden) ||
    !dense_shape(value_projection, hidden, hidden) ||
    !dense_shape(attention_output_projection, hidden, hidden) ||
    video_mlp.up.input_width != hidden ||
    video_mlp.down.output_width != hidden ||
    audio_mlp.up.input_width != hidden ||
    audio_mlp.down.output_width != hidden ||
    !dense_shape(video_output_projection, hidden, video_width) ||
    !dense_shape(audio_output_projection, hidden, audio_width) {
    raise InvalidWeightShape
  }
  {
    plan_digest: plan.plan_digest(),
    text_projection,
    keyframe_projection,
    reference_projection,
    video_input_projection,
    audio_input_projection,
    query_projection,
    key_projection,
    value_projection,
    attention_output_projection,
    video_mlp,
    audio_mlp,
    video_output_projection,
    audio_output_projection,
  }
}
