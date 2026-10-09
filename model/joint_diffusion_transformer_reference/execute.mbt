///|
fn require_finite(
  value : Double,
) -> Double raise JointTinyTransformerReferenceError {
  if value.is_nan() || value.is_inf() {
    raise NonFiniteComputation
  }
  value
}

///|
fn require_elements(
  count : Int64,
  limits : JointTinyTransformerLimits,
) -> Int raise JointTinyTransformerReferenceError {
  if count < 0L || count > limits.maximum_elements.to_int64() {
    raise ElementBudgetExceeded
  }
  count.to_int()
}

///|
fn dense_row(
  dense : JointTinyDense,
  input : ArrayView[Double],
) -> Array[Double] raise JointTinyTransformerReferenceError {
  if input.length() != dense.input_width {
    raise InvalidDenseShape
  }
  let output : Array[Double] = []
  for output_index in 0..<dense.output_width {
    let mut value = dense.bias[output_index]
    for input_index, item in input {
      value = require_finite(
        value +
        dense.weights[output_index * dense.input_width + input_index] * item,
      )
    }
    output.push(value)
  }
  output
}

///|
fn project_conditioning(
  matrix : JointTinyConditioningMatrix,
  dense : JointTinyDense,
  limits : JointTinyTransformerLimits,
) -> Array[Double] raise JointTinyTransformerReferenceError {
  if matrix.width != dense.input_width {
    raise InvalidConditioningShape(matrix.source)
  }
  ignore(
    require_elements(
      matrix.rows.to_int64() * dense.output_width.to_int64(),
      limits,
    ),
  )
  let output : Array[Double] = []
  for row in 0..<matrix.rows {
    let start = row * matrix.width
    for value in dense_row(dense, matrix.values[start:start + matrix.width]) {
      output.push(value)
    }
  }
  output
}

///|
fn project_flat_rows(
  values : ArrayView[Double],
  rows : Int,
  dense : JointTinyDense,
  limits : JointTinyTransformerLimits,
) -> Array[Double] raise JointTinyTransformerReferenceError {
  if values.length().to_int64() !=
    rows.to_int64() * dense.input_width.to_int64() {
    raise InvalidRequest
  }
  ignore(
    require_elements(rows.to_int64() * dense.output_width.to_int64(), limits),
  )
  let output : Array[Double] = []
  for row in 0..<rows {
    let start = row * dense.input_width
    for value in dense_row(dense, values[start:start + dense.input_width]) {
      output.push(value)
    }
  }
  output
}

///|
fn has_stage(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  kind : @joint_diffusion_plan.JointDiffusionStageKind,
) -> Bool {
  plan.stages().any(stage => stage.kind() == kind)
}

///|
fn validate_conditioning_contract(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  conditioning : JointTinyConditioningInput,
) -> Unit raise JointTinyTransformerReferenceError {
  if conditioning.text.width != plan.denoiser().text_conditioning_dimension() {
    raise InvalidConditioningShape(TextConditioning)
  }
  if conditioning.keyframe.rows > 0 && conditioning.reference.rows > 0 {
    raise InvalidConditioningContract
  }
  if conditioning.keyframe.rows > 0 &&
    (
      !has_stage(plan, EncodeKeyframeConditioning) ||
      plan.conditioning().max_keyframes() <= 0
    ) {
    raise InvalidConditioningContract
  }
  if conditioning.reference.rows > 0 &&
    (
      !has_stage(plan, EncodeReferenceConditioning) ||
      plan.conditioning().max_total_references() <= 0
    ) {
    raise InvalidConditioningContract
  }
}

///|
fn attention(
  packed : ArrayView[Double],
  rows : Int,
  hidden : Int,
  heads : Int,
  weights : JointTinyTransformerWeights,
  limits : JointTinyTransformerLimits,
) -> Array[Double] raise JointTinyTransformerReferenceError {
  let head_width = hidden / heads
  let q = project_flat_rows(packed, rows, weights.query_projection, limits)
  let k = project_flat_rows(packed, rows, weights.key_projection, limits)
  let v = project_flat_rows(packed, rows, weights.value_projection, limits)
  ignore(require_elements(rows.to_int64() * rows.to_int64(), limits))
  let mixed = Array::make(rows * hidden, 0.0)
  let scale = head_width.to_double().sqrt()
  for query_row in 0..<rows {
    for head in 0..<heads {
      let scores : Array[Double] = []
      let mut maximum = -1.7976931348623157e308
      for key_row in 0..<rows {
        let mut dot = 0.0
        for lane in 0..<head_width {
          let offset = head * head_width + lane
          dot = require_finite(
            dot + q[query_row * hidden + offset] * k[key_row * hidden + offset],
          )
        }
        let score = require_finite(dot / scale)
        scores.push(score)
        if score > maximum {
          maximum = score
        }
      }
      let mut denominator = 0.0
      for index in 0..<scores.length() {
        scores[index] = require_finite(@math.exp(scores[index] - maximum))
        denominator = require_finite(denominator + scores[index])
      }
      if denominator <= 0.0 {
        raise NonFiniteComputation
      }
      for lane in 0..<head_width {
        let offset = head * head_width + lane
        let mut value = 0.0
        for key_row in 0..<rows {
          value = require_finite(
            value + scores[key_row] / denominator * v[key_row * hidden + offset],
          )
        }
        mixed[query_row * hidden + offset] = value
      }
    }
  }
  let attended : Array[Double] = []
  for row in 0..<rows {
    let start = row * hidden
    let projected = dense_row(
      weights.attention_output_projection,
      mixed[start:start + hidden],
    )
    for lane in 0..<hidden {
      attended.push(require_finite(packed[start + lane] + projected[lane]))
    }
  }
  attended
}

///|
fn mlp_residual(
  input : ArrayView[Double],
  mlp : JointTinyMlp,
) -> Array[Double] raise JointTinyTransformerReferenceError {
  let activated = dense_row(mlp.up, input)
  for index in 0..<activated.length() {
    if activated[index] < 0.0 {
      activated[index] = 0.0
    }
  }
  let projected = dense_row(mlp.down, activated)
  let output : Array[Double] = []
  for index, value in input {
    output.push(require_finite(value + projected[index]))
  }
  output
}

///|
fn modality_prediction(
  attended : ArrayView[Double],
  row_start : Int,
  rows : Int,
  hidden : Int,
  mlp : JointTinyMlp,
  output_projection : JointTinyDense,
  limits : JointTinyTransformerLimits,
) -> Array[Double] raise JointTinyTransformerReferenceError {
  ignore(
    require_elements(
      rows.to_int64() * output_projection.output_width.to_int64(),
      limits,
    ),
  )
  let output : Array[Double] = []
  for row in 0..<rows {
    let start = (row_start + row) * hidden
    let transformed = mlp_residual(attended[start:start + hidden], mlp)
    for value in dense_row(output_projection, transformed) {
      output.push(value)
    }
  }
  output
}

///|
fn rectified_flow_update(
  latent : ArrayView[Double],
  prediction : ArrayView[Double],
  sigma : Double,
  next_sigma : Double,
) -> Array[Double] raise JointTinyTransformerReferenceError {
  if latent.length() != prediction.length() {
    raise InvalidRequest
  }
  // H3 clean-state velocity convention: x0 = x + sigma * v. This
  // independent Double oracle intentionally does not call the production F32
  // coefficient/update code. The equivalent Euler delta is sigma-next_sigma,
  // not the negative delta used by noise-prediction parameterizations.
  if sigma <= 0.0 || next_sigma < 0.0 || next_sigma > sigma {
    raise InvalidRequest
  }
  let ratio = require_finite(next_sigma / sigma)
  let output : Array[Double] = []
  for index, value in latent {
    let clean = require_finite(value + sigma * prediction[index])
    output.push(require_finite(ratio * value + (1.0 - ratio) * clean))
  }
  output
}

///|
/// Executes one bounded scalar reference transformer evaluation and one
/// modality-specific shifted Rectified Flow Euler update.
pub fn run_tiny_joint_transformer_step(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  conditioning : JointTinyConditioningInput,
  weights : JointTinyTransformerWeights,
  seed~ : Int64,
  step_index~ : Int,
  limits~ : JointTinyTransformerLimits,
) -> JointTinyDenoiseStepResult raise JointTinyTransformerReferenceError {
  validate_conditioning_contract(plan, conditioning)
  if weights.plan_digest != plan.plan_digest() {
    raise InvalidPlan
  }
  let conditioning_rows = conditioning.text.rows +
    conditioning.keyframe.rows +
    conditioning.reference.rows
  let layout = @joint_diffusion_reference.JointPackedRowLayout::new(
    plan,
    request,
    conditioning_rows~,
  ) catch {
    _ => raise InvalidRequest
  }
  let hidden = layout.hidden_width()
  let heads = plan.denoiser().attention_head_count()
  let ff = weights.video_mlp.up.output_width
  if layout.total_rows() > limits.maximum_rows ||
    hidden > limits.maximum_hidden_width ||
    ff > limits.maximum_feed_forward_width ||
    weights.audio_mlp.up.output_width > limits.maximum_feed_forward_width ||
    heads <= 1 ||
    hidden % heads != 0 {
    raise InvalidPlan
  }
  let packed_elements = require_elements(
    layout.total_rows().to_int64() * hidden.to_int64(),
    limits,
  )
  let conditioning_draws = conditioning.text.values.length() +
    conditioning.keyframe.values.length() +
    conditioning.reference.values.length()
  let rng = @joint_diffusion_reference.CounterBasedJointLatentRng::new(
    plan,
    request,
    seed~,
    conditioning_draw_limit=conditioning_draws.to_int64(),
  ) catch {
    _ => raise InvalidRequest
  }
  ignore(require_elements(rng.draw_limit(VideoLatentStream), limits))
  ignore(require_elements(rng.draw_limit(AudioLatentStream), limits))
  let initial_video : Array[Double] = []
  for counter in 0L..<rng.draw_limit(VideoLatentStream) {
    initial_video.push(
      rng.normal(VideoLatentStream, counter) catch {
        _ => raise InvalidRequest
      },
    )
  }
  let initial_audio : Array[Double] = []
  for counter in 0L..<rng.draw_limit(AudioLatentStream) {
    initial_audio.push(
      rng.normal(AudioLatentStream, counter) catch {
        _ => raise InvalidRequest
      },
    )
  }
  let projected_conditioning : Array[Double] = []
  for
    value in project_conditioning(
      conditioning.text,
      weights.text_projection,
      limits,
    ) {
    projected_conditioning.push(value)
  }
  for
    value in project_conditioning(
      conditioning.keyframe,
      weights.keyframe_projection,
      limits,
    ) {
    projected_conditioning.push(value)
  }
  for
    value in project_conditioning(
      conditioning.reference,
      weights.reference_projection,
      limits,
    ) {
    projected_conditioning.push(value)
  }
  let projected_video = project_flat_rows(
    initial_video,
    layout.video_rows(),
    weights.video_input_projection,
    limits,
  )
  let projected_audio = project_flat_rows(
    initial_audio,
    layout.audio_rows(),
    weights.audio_input_projection,
    limits,
  )
  let packed = @joint_diffusion_reference.recombine_projected_rows(
    layout,
    projected_conditioning,
    projected_video,
    projected_audio,
    max_elements=packed_elements,
  ) catch {
    _ => raise InvalidRequest
  }
  let attended = attention(
    packed.values(),
    layout.total_rows(),
    hidden,
    heads,
    weights,
    limits,
  )
  let video_prediction = modality_prediction(
    attended,
    layout.row_start(VideoRows),
    layout.video_rows(),
    hidden,
    weights.video_mlp,
    weights.video_output_projection,
    limits,
  )
  let audio_prediction = modality_prediction(
    attended,
    layout.row_start(AudioRows),
    layout.audio_rows(),
    hidden,
    weights.audio_mlp,
    weights.audio_output_projection,
    limits,
  )
  let video_schedule = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
    plan,
    request,
    VideoLatent,
  ) catch {
    _ => raise InvalidRequest
  }
  let audio_schedule = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
    plan,
    request,
    AudioLatent,
  ) catch {
    _ => raise InvalidRequest
  }
  if step_index < 0 ||
    step_index >= video_schedule.evaluation_count() ||
    step_index >= audio_schedule.evaluation_count() {
    raise InvalidStepIndex(step_index)
  }
  guard video_schedule.sigma(step_index) is Some(video_sigma) &&
    video_schedule.sigma(step_index + 1) is Some(video_next_sigma) &&
    audio_schedule.sigma(step_index) is Some(audio_sigma) &&
    audio_schedule.sigma(step_index + 1) is Some(audio_next_sigma) else {
    raise InvalidStepIndex(step_index)
  }
  {
    plan_digest: plan.plan_digest(),
    step_index,
    initial_video: ReadOnlyArray::from_array(initial_video),
    initial_audio: ReadOnlyArray::from_array(initial_audio),
    packed_projected: packed.values(),
    attended: ReadOnlyArray::from_array(attended),
    video_prediction: ReadOnlyArray::from_array(video_prediction),
    audio_prediction: ReadOnlyArray::from_array(audio_prediction),
    updated_video: ReadOnlyArray::from_array(
      rectified_flow_update(
        initial_video, video_prediction, video_sigma, video_next_sigma,
      ),
    ),
    updated_audio: ReadOnlyArray::from_array(
      rectified_flow_update(
        initial_audio, audio_prediction, audio_sigma, audio_next_sigma,
      ),
    ),
    video_sigma,
    video_next_sigma,
    audio_sigma,
    audio_next_sigma,
  }
}
