///|
fn test_content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn tiny_transformer_plan(
  reference? : Bool = false,
) -> @joint_diffusion_plan.JointDiffusionPlan {
  let stages = if reference {
    [
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=ResolveRequest,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeTextConditioning,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeReferenceConditioning,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=InitializeJointLatents,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=JointAudioVideoDenoise,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeVideo,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeAudio,
        presence=Required,
      ),
    ]
  } else {
    [
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=ResolveRequest,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeTextConditioning,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeKeyframeConditioning,
        presence=Optional,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=InitializeJointLatents,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=JointAudioVideoDenoise,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeVideo,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeAudio,
        presence=Required,
      ),
    ]
  }
  try! @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=1,
    content_digest=test_content(if reference { '7' } else { '6' }),
    workflows=if reference {
      [ReferenceToAudioVideo]
    } else {
      [TextToAudioVideo, FirstLastFrameToAudioVideo]
    },
    stages~,
    video_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=VideoLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=2,
      shift_denominator=1,
    ),
    audio_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=AudioLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=1,
      shift_denominator=1,
    ),
    shapes=@joint_diffusion_plan.JointDiffusionShapeContract::new(
      frames_per_second=4,
      min_duration_seconds=1,
      max_duration_seconds=2,
      frames_per_chunk=3,
      latents_per_chunk=1,
      canvas_multiple=4,
      max_canvas_pixels=16L,
      min_aspect_numerator=1,
      min_aspect_denominator=1,
      max_aspect_numerator=1,
      max_aspect_denominator=1,
      video_latent_channels=2,
      video_spatial_compression=2,
      audio_latent_channels=2,
      audio_latents_per_second=2,
      audio_sample_rate=8,
      audio_output_channels=1,
      audio_vae_latent_dimension=4,
      audio_vae_decoder_dimension=2,
      min_inference_steps=2,
      max_inference_steps=8,
      video_patch_temporal=1,
      video_patch_height=2,
      video_patch_width=2,
    ),
    conditioning=@joint_diffusion_plan.JointDiffusionConditioningLimits::new(
      max_keyframes=if reference { 0 } else { 2 },
      max_image_references=if reference { 2 } else { 0 },
      max_video_references=0,
      max_audio_references=0,
      max_total_references=if reference { 2 } else { 0 },
      allow_audio_only_references=false,
    ),
    projection=@joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    denoiser=@joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=4,
      layer_count=1,
      refiner_layer_count=0,
      attention_head_count=2,
      attention_head_dimension=2,
      feed_forward_dimension=1,
      text_conditioning_dimension=4,
      timestep_frequency_dimension=2,
    ),
  )
}

///|
fn tiny_request(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  try! plan
  .shapes()
  .resolve_request(height=4, width=4, requested_frames=4, inference_steps=5)
}

///|
fn dense_bias(input : Int, bias : ArrayView[Double]) -> JointTinyDense {
  try! JointTinyDense::new(
    input_width=input,
    output_width=bias.length(),
    weights=Array::make(input * bias.length(), 0.0),
    bias~,
  )
}

///|
fn identity_dense(width : Int) -> JointTinyDense {
  let values = Array::make(width * width, 0.0)
  for index in 0..<width {
    values[index * width + index] = 1.0
  }
  try! JointTinyDense::new(
    input_width=width,
    output_width=width,
    weights=values,
    bias=Array::make(width, 0.0),
  )
}

///|
fn output_selector(width : Int) -> JointTinyDense {
  let values = Array::make(width * 4, 0.0)
  for output in 0..<width {
    values[output * 4 + output % 4] = 1.0
  }
  try! JointTinyDense::new(
    input_width=4,
    output_width=width,
    weights=values,
    bias=Array::make(width, 0.0),
  )
}

///|
fn tiny_weights(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
) -> JointTinyTransformerWeights {
  let zero4 = dense_bias(4, [0.0, 0.0, 0.0, 0.0])
  let up = dense_bias(4, [1.0])
  let video_down = try! JointTinyDense::new(
    input_width=1,
    output_width=4,
    weights=[1.0, 0.0, 0.0, 0.0],
    bias=[0.0, 0.0, 0.0, 0.0],
  )
  let audio_down = try! JointTinyDense::new(
    input_width=1,
    output_width=4,
    weights=[0.0, 1.0, 0.0, 0.0],
    bias=[0.0, 0.0, 0.0, 0.0],
  )
  try! JointTinyTransformerWeights::new(
    plan,
    keyframe_input_width=2,
    reference_input_width=3,
    text_projection=dense_bias(4, [1.0, 0.0, 0.0, 0.0]),
    keyframe_projection=dense_bias(2, [0.0, 1.0, 0.0, 0.0]),
    reference_projection=dense_bias(3, [0.0, 0.0, 1.0, 0.0]),
    video_input_projection=dense_bias(8, [0.0, 0.0, 0.0, 1.0]),
    audio_input_projection=dense_bias(2, [1.0, 1.0, 0.0, 0.0]),
    query_projection=zero4,
    key_projection=zero4,
    value_projection=identity_dense(4),
    attention_output_projection=identity_dense(4),
    video_mlp=try! JointTinyMlp::new(up~, down=video_down),
    audio_mlp=try! JointTinyMlp::new(up~, down=audio_down),
    video_output_projection=output_selector(8),
    audio_output_projection=output_selector(2),
  )
}

///|
fn conditioning(reference : Bool) -> JointTinyConditioningInput {
  try! JointTinyConditioningInput::new(
    text=try! JointTinyConditioningMatrix::new(
      source=TextConditioning,
      rows=1,
      width=4,
      values=[2.0, 3.0, 4.0, 5.0],
    ),
    keyframe=try! JointTinyConditioningMatrix::new(
      source=KeyframeConditioning,
      rows=if reference { 0 } else { 1 },
      width=2,
      values=if reference { [] } else { [6.0, 7.0] },
    ),
    reference=try! JointTinyConditioningMatrix::new(
      source=ReferenceConditioning,
      rows=if reference { 1 } else { 0 },
      width=3,
      values=if reference { [8.0, 9.0, 10.0] } else { [] },
    ),
  )
}

///|
fn limits(
  maximum_rows? : Int = 16,
  maximum_elements? : Int = 1024,
) -> JointTinyTransformerLimits {
  try! JointTinyTransformerLimits::new(
    maximum_rows~,
    maximum_hidden_width=8,
    maximum_feed_forward_width=8,
    maximum_elements~,
  )
}

///|
fn approximately(actual : Double, expected : Double) -> Bool {
  (actual - expected).abs() < 0.000000000001
}

///|
test "FL2VA keyframe path matches independently calculated attention and Euler fixture" {
  let plan = tiny_transformer_plan()
  let result = run_tiny_joint_transformer_step(
    plan,
    tiny_request(plan),
    conditioning(false),
    tiny_weights(plan),
    seed=42L,
    step_index=0,
    limits=limits(),
  )
  assert_eq(result.packed_projected().length(), 28)
  assert_eq(result.video_prediction().length(), 24)
  assert_eq(result.audio_prediction().length(), 4)
  let video_expected = [10.0 / 7.0, 3.0 / 7.0, 0.0, 10.0 / 7.0]
  for index in 0..<8 {
    assert_true(
      approximately(result.video_prediction()[index], video_expected[index % 4]),
    )
  }
  assert_true(approximately(result.audio_prediction()[0], 10.0 / 7.0))
  assert_true(approximately(result.audio_prediction()[1], 17.0 / 7.0))
  assert_true(approximately(result.video_sigma(), 1.0))
  assert_true(approximately(result.video_next_sigma(), 1.5005 / 1.75025))
  assert_true(approximately(result.audio_sigma(), 1.0))
  assert_true(approximately(result.audio_next_sigma(), 0.75025))
  for index in 0..<result.updated_video().length() {
    let expected = result.initial_video()[index] +
      (1.0 - result.video_next_sigma()) * result.video_prediction()[index]
    assert_true(approximately(result.updated_video()[index], expected))
  }
}

///|
test "both modalities use clean-state velocity sign at every step including terminal" {
  for reference in [false, true] {
    let plan = tiny_transformer_plan(reference~)
    let request = tiny_request(plan)
    for step_index in 0..<request.model_evaluations() {
      let result = run_tiny_joint_transformer_step(
        plan,
        request,
        conditioning(reference),
        tiny_weights(plan),
        seed=42L,
        step_index~,
        limits=limits(),
      )
      for
        (initial, velocity, updated, sigma, next_sigma) in [
          (
            result.initial_video(),
            result.video_prediction(),
            result.updated_video(),
            result.video_sigma(),
            result.video_next_sigma(),
          ),
          (
            result.initial_audio(),
            result.audio_prediction(),
            result.updated_audio(),
            result.audio_sigma(),
            result.audio_next_sigma(),
          ),
        ] {
        for i in 0..<initial.length() {
          // Independently simplified Euler form rather than implementation's
          // clean-state blend; detects the old reversed velocity sign.
          let expected = initial[i] + (sigma - next_sigma) * velocity[i]
          assert_true(approximately(updated[i], expected))
          if velocity[i] > 0.0 {
            assert_true(updated[i] > initial[i])
          }
          if step_index == request.model_evaluations() - 1 {
            assert_eq(next_sigma, 0.0)
            assert_true(
              approximately(updated[i], initial[i] + sigma * velocity[i]),
            )
          }
        }
      }
    }
  }
}

///|
test "Ref2VA reference projection changes packed attention and modality MLP outputs" {
  let plan = tiny_transformer_plan(reference=true)
  let result = run_tiny_joint_transformer_step(
    plan,
    tiny_request(plan),
    conditioning(true),
    tiny_weights(plan),
    seed=42L,
    step_index=0,
    limits=limits(),
  )
  let video_expected = [10.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 10.0 / 7.0]
  for index in 0..<8 {
    assert_true(
      approximately(result.video_prediction()[index], video_expected[index % 4]),
    )
  }
  assert_true(approximately(result.audio_prediction()[0], 10.0 / 7.0))
  assert_true(approximately(result.audio_prediction()[1], 16.0 / 7.0))
  assert_true(result.initial_video() != result.updated_video())
  assert_true(result.initial_audio() != result.updated_audio())
}

///|
test "official MiniMax variants expose only the tested generic conditioning stages" {
  fn official(
    variant : @minimax_h3_spec.MiniMaxH3Variant,
  ) -> @joint_diffusion_plan.JointDiffusionPlan {
    try! @minimax_h3.build(
      @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
        @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
          variant~,
          checkpoint_format=DiffusersV1,
        ),
        test_content('5'),
      ),
    )
  }
  assert_true(
    official(Fl2va)
    .stages()
    .any(stage => stage.kind() == EncodeKeyframeConditioning),
  )
  assert_true(
    official(Ref2va)
    .stages()
    .any(stage => stage.kind() == EncodeReferenceConditioning),
  )
}

///|
test "shape finite step and allocation bounds fail closed" {
  try
    JointTinyConditioningMatrix::new(source=TextConditioning, rows=1, width=4, values=[
      0.0,
      0.0,
      0.0,
      1.0 / 0.0,
    ])
  catch {
    NonFiniteInput => ()
    _ => fail("unexpected nonfinite error")
  } noraise {
    _ => fail("nonfinite conditioning admitted")
  }
  let plan = tiny_transformer_plan()
  let both = try! JointTinyConditioningInput::new(
    text=try! JointTinyConditioningMatrix::new(
      source=TextConditioning,
      rows=1,
      width=4,
      values=[0.0, 0.0, 0.0, 0.0],
    ),
    keyframe=try! JointTinyConditioningMatrix::new(
      source=KeyframeConditioning,
      rows=1,
      width=2,
      values=[0.0, 0.0],
    ),
    reference=try! JointTinyConditioningMatrix::new(
      source=ReferenceConditioning,
      rows=1,
      width=3,
      values=[0.0, 0.0, 0.0],
    ),
  )
  try
    run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      both,
      tiny_weights(plan),
      seed=42L,
      step_index=0,
      limits=limits(),
    )
  catch {
    InvalidConditioningContract => ()
    _ => fail("unexpected mixed conditioning error")
  } noraise {
    _ => fail("keyframe and reference conditioning were mixed")
  }
  try
    run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      conditioning(false),
      tiny_weights(plan),
      seed=42L,
      step_index=5,
      limits=limits(),
    )
  catch {
    InvalidStepIndex(5) => ()
    _ => fail("unexpected step error")
  } noraise {
    _ => fail("terminal step admitted")
  }
  try
    run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      conditioning(false),
      tiny_weights(plan),
      seed=42L,
      step_index=0,
      limits=limits(maximum_rows=6),
    )
  catch {
    InvalidPlan => ()
    _ => fail("unexpected row bound error")
  } noraise {
    _ => fail("row bound exceeded")
  }
  try
    run_tiny_joint_transformer_step(
      plan,
      tiny_request(plan),
      conditioning(false),
      tiny_weights(plan),
      seed=42L,
      step_index=0,
      limits=limits(maximum_elements=27),
    )
  catch {
    ElementBudgetExceeded => ()
    _ => fail("unexpected element bound error")
  } noraise {
    _ => fail("element budget exceeded")
  }
}
