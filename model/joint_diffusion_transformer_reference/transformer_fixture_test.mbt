///|
fn test_content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn tiny_transformer_plan(
  reference~ : Bool = false,
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
fn dense_bias(
  input : Int32,
  bias : ArrayView[Double],
) -> @joint_diffusion_transformer_reference.JointTinyDense {
  try! @joint_diffusion_transformer_reference.JointTinyDense::new(
    input_width=input,
    output_width=bias.length(),
    weights=Array::make(input * bias.length(), 0.0),
    bias~,
  )
}

///|
fn identity_dense(
  width : Int32,
) -> @joint_diffusion_transformer_reference.JointTinyDense {
  let values = Array::make(width * width, 0.0)
  for index in 0..<width {
    values[index * width + index] = 1.0
  }
  try! @joint_diffusion_transformer_reference.JointTinyDense::new(
    input_width=width,
    output_width=width,
    weights=values,
    bias=Array::make(width, 0.0),
  )
}

///|
fn output_selector(
  width : Int32,
) -> @joint_diffusion_transformer_reference.JointTinyDense {
  let values = Array::make(width * 4, 0.0)
  for output in 0..<width {
    values[output * 4 + output % 4] = 1.0
  }
  try! @joint_diffusion_transformer_reference.JointTinyDense::new(
    input_width=4,
    output_width=width,
    weights=values,
    bias=Array::make(width, 0.0),
  )
}

///|
fn tiny_weights(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
) -> @joint_diffusion_transformer_reference.JointTinyTransformerWeights {
  let zero4 = dense_bias(4, [0.0, 0.0, 0.0, 0.0])
  let up = dense_bias(4, [1.0])
  let video_down = try! @joint_diffusion_transformer_reference.JointTinyDense::new(
    input_width=1,
    output_width=4,
    weights=[1.0, 0.0, 0.0, 0.0],
    bias=[0.0, 0.0, 0.0, 0.0],
  )
  let audio_down = try! @joint_diffusion_transformer_reference.JointTinyDense::new(
    input_width=1,
    output_width=4,
    weights=[0.0, 1.0, 0.0, 0.0],
    bias=[0.0, 0.0, 0.0, 0.0],
  )
  try! @joint_diffusion_transformer_reference.JointTinyTransformerWeights::new(
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
    video_mlp=try! @joint_diffusion_transformer_reference.JointTinyMlp::new(
      up~,
      down=video_down,
    ),
    audio_mlp=try! @joint_diffusion_transformer_reference.JointTinyMlp::new(
      up~,
      down=audio_down,
    ),
    video_output_projection=output_selector(8),
    audio_output_projection=output_selector(2),
  )
}

///|
fn conditioning(
  reference : Bool,
) -> @joint_diffusion_transformer_reference.JointTinyConditioningInput {
  try! @joint_diffusion_transformer_reference.JointTinyConditioningInput::new(
    text=try! @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=TextConditioning,
      rows=1,
      width=4,
      values=[2.0, 3.0, 4.0, 5.0],
    ),
    keyframe=try! @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=KeyframeConditioning,
      rows=if reference { 0 } else { 1 },
      width=2,
      values=if reference { [] } else { [6.0, 7.0] },
    ),
    reference=try! @joint_diffusion_transformer_reference.JointTinyConditioningMatrix::new(
      source=ReferenceConditioning,
      rows=if reference { 1 } else { 0 },
      width=3,
      values=if reference { [8.0, 9.0, 10.0] } else { [] },
    ),
  )
}

///|
fn limits(
  maximum_rows~ : Int32 = 16,
  maximum_elements~ : Int32 = 1024,
) -> @joint_diffusion_transformer_reference.JointTinyTransformerLimits {
  try! @joint_diffusion_transformer_reference.JointTinyTransformerLimits::new(
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
