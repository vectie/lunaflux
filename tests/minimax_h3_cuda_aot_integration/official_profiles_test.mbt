///|
fn official_requirements(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements {
  let metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant~,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(String::make(64, '6')),
  )
  try! @minimax_h3_kernel_requirements.project(metadata)
}

///|
fn shifted_flow_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      ShiftedFlowStep(actual, _, _, _) if actual == latent => return ordinal
      _ => ()
    }
  }
  abort("official MiniMax profile lost a shifted-flow requirement")
}

///|
fn projection_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoLatentProjection(_, _, _, _, _) if latent == VideoLatent =>
        return ordinal
      AudioLatentProjection(_, _) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("official MiniMax profile lost a latent-projection requirement")
}

///|
fn output_projection_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoOutputProjection(_, _, _, _, _) if latent == VideoLatent =>
        return ordinal
      AudioOutputProjection(_, _) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("official MiniMax profile lost an output-projection requirement")
}

///|
fn final_norm_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoFinalRmsNorm(_, _, _) if latent == VideoLatent => return ordinal
      AudioFinalRmsNorm(_, _, _) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("official MiniMax profile lost a final-RMSNorm requirement")
}

///|
fn final_adaln_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoFinalAdaLnF32Boundary(_) if latent == VideoLatent => return ordinal
      AudioFinalAdaLnF32Boundary(_) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("official MiniMax profile lost a final-AdaLN requirement")
}

///|
fn parameter_row_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      FinalAdaLnParameterRow(_, _) => return ordinal
      _ => ()
    }
  }
  abort("official MiniMax profile lost final AdaLN parameter row")
}

///|
fn packed_gather_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      PackedModalityAdaLnRows(_) => return ordinal
      _ => ()
    }
  }
  abort("packed gather missing")
}

///|
fn timestep_frequency_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      TimestepFrequencyRows(_) => return ordinal
      _ => ()
    }
  }
  abort("timestep frequency rows missing")
}

///|
fn integration_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn integration_compiler() -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, '5'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn integration_limits() -> @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits {
  try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=300000000L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
test "official MiniMax H3 profiles lower exact projection and flow geometry" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  let recipe_digests : Array[
    @joint_diffusion_cuda_aot.JointDiffusionCudaRecipeDigest,
  ] = []
  for variant in variants {
    let requirements = official_requirements(variant)
    let (timestep_selection, modality_selection, timestep_values) = packed_selection_fixture(
      variant,
    )
    let request = try! requirements
      .shape_contract()
      .resolve_request(
        height=768,
        width=1344,
        requested_frames=120,
        inference_steps=50,
      )
    assert_eq(request.frames(), 124)
    let video = @joint_diffusion_cuda_aot.lower_shifted_flow_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=shifted_flow_ordinal(requirements, VideoLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(video.element_count(), 3580416L)
    assert_eq(video.dimensions().grid_x(), 13986)
    assert_eq(video.operands()[0].byte_count(), 14321664L)
    assert_eq(video.operands()[1].byte_count(), 14321664L)
    assert_eq(video.operands()[2].byte_count(), 12L)
    assert_eq(video.operands()[3].byte_count(), 14321664L)
    let audio = @joint_diffusion_cuda_aot.lower_shifted_flow_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=shifted_flow_ordinal(requirements, AudioLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(audio.element_count(), 13248L)
    assert_eq(audio.dimensions().grid_x(), 52)
    assert_eq(audio.operands()[0].byte_count(), 52992L)
    assert_eq(audio.operands()[2].byte_count(), 12L)
    assert_false(video.manifest_bindable())
    assert_false(audio.manifest_bindable())
    let video_projection = @joint_diffusion_cuda_aot.lower_latent_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=projection_ordinal(requirements, VideoLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(video_projection.row_count(), 37296L)
    assert_eq(video_projection.input_width(), 96)
    assert_eq(video_projection.output_width(), 5376)
    assert_eq(video_projection.input_element_count(), 3580416L)
    assert_eq(video_projection.output_element_count(), 200503296L)
    assert_eq(video_projection.model_weight_element_count(), 521472L)
    assert_eq(video_projection.projection_weight_element_count(), 516096L)
    assert_eq(video_projection.projection_bias_element_count(), 5376L)
    assert_eq(video_projection.dimensions().grid_x(), 783216)
    assert_eq(video_projection.operands()[0].byte_count(), 14321664L)
    assert_eq(video_projection.operands()[1].byte_count(), 2064384L)
    assert_eq(video_projection.operands()[2].byte_count(), 21504L)
    assert_eq(video_projection.operands()[3].byte_count(), 401006592L)
    let audio_projection = @joint_diffusion_cuda_aot.lower_latent_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=projection_ordinal(requirements, AudioLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(audio_projection.row_count(), 414L)
    assert_eq(audio_projection.input_width(), 32)
    assert_eq(audio_projection.output_width(), 5376)
    assert_eq(audio_projection.input_element_count(), 13248L)
    assert_eq(audio_projection.output_element_count(), 2225664L)
    assert_eq(audio_projection.model_weight_element_count(), 177408L)
    assert_eq(audio_projection.projection_weight_element_count(), 172032L)
    assert_eq(audio_projection.projection_bias_element_count(), 5376L)
    assert_eq(audio_projection.dimensions().grid_x(), 8694)
    assert_eq(audio_projection.operands()[0].byte_count(), 52992L)
    assert_eq(audio_projection.operands()[1].byte_count(), 688128L)
    assert_eq(audio_projection.operands()[2].byte_count(), 21504L)
    assert_eq(audio_projection.operands()[3].byte_count(), 4451328L)
    assert_false(video_projection.manifest_bindable())
    assert_false(audio_projection.manifest_bindable())
    let video_norm = @joint_diffusion_cuda_aot.lower_final_rms_norm_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=final_norm_ordinal(requirements, VideoLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(video_norm.operation(), VideoFinalRmsNorm(5376, 1, 100000))
    assert_eq(video_norm.capability(), VideoFinalRmsNormAot)
    assert_eq(video_norm.row_count(), 37296L)
    assert_eq(video_norm.hidden_size(), 5376)
    assert_eq(video_norm.element_count(), 200503296L)
    assert_eq(video_norm.weight_element_count(), 5376L)
    assert_eq(video_norm.dimensions().grid_x(), 37296)
    assert_eq(video_norm.operands()[0].role(), VideoFinalNormInput)
    assert_eq(video_norm.operands()[0].byte_count(), 401006592L)
    assert_eq(video_norm.operands()[1].role(), FinalNormWeight)
    assert_eq(video_norm.operands()[1].byte_count(), 10752L)
    assert_eq(video_norm.operands()[2].role(), VideoFinalNormOutput)
    assert_eq(video_norm.operands()[2].byte_count(), 401006592L)
    assert_false(video_norm.manifest_bindable())
    let audio_norm = @joint_diffusion_cuda_aot.lower_final_rms_norm_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=final_norm_ordinal(requirements, AudioLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(audio_norm.operation(), AudioFinalRmsNorm(5376, 1, 100000))
    assert_eq(audio_norm.capability(), AudioFinalRmsNormAot)
    assert_eq(audio_norm.row_count(), 414L)
    assert_eq(audio_norm.element_count(), 2225664L)
    assert_eq(audio_norm.dimensions().grid_x(), 414)
    assert_eq(audio_norm.operands()[0].role(), AudioFinalNormInput)
    assert_eq(audio_norm.operands()[0].byte_count(), 4451328L)
    assert_eq(audio_norm.operands()[1].byte_count(), 10752L)
    assert_eq(audio_norm.operands()[2].role(), AudioFinalNormOutput)
    assert_eq(audio_norm.operands()[2].byte_count(), 4451328L)
    assert_false(audio_norm.manifest_bindable())
    let video_adaln = @joint_diffusion_cuda_aot.lower_final_adaln_f32_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=final_adaln_ordinal(requirements, VideoLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(video_adaln.operation(), VideoFinalAdaLnF32Boundary(5376))
    assert_eq(video_adaln.row_count(), 37296L)
    assert_eq(video_adaln.element_count(), 200503296L)
    assert_eq(video_adaln.dimensions().grid_x(), 783216)
    assert_eq(video_adaln.operands()[0].byte_count(), 401006592L)
    assert_eq(video_adaln.operands()[1].byte_count(), 401006592L)
    assert_eq(video_adaln.operands()[2].byte_count(), 401006592L)
    assert_eq(video_adaln.operands()[3].role(), VideoF32HeadInput)
    assert_eq(video_adaln.operands()[3].byte_count(), 802013184L)
    assert_false(video_adaln.manifest_bindable())
    let audio_adaln = @joint_diffusion_cuda_aot.lower_final_adaln_f32_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=final_adaln_ordinal(requirements, AudioLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(audio_adaln.operation(), AudioFinalAdaLnF32Boundary(5376))
    assert_eq(audio_adaln.row_count(), 414L)
    assert_eq(audio_adaln.element_count(), 2225664L)
    assert_eq(audio_adaln.dimensions().grid_x(), 8694)
    assert_eq(audio_adaln.operands()[0].byte_count(), 4451328L)
    assert_eq(audio_adaln.operands()[3].role(), AudioF32HeadInput)
    assert_eq(audio_adaln.operands()[3].byte_count(), 8902656L)
    assert_false(audio_adaln.manifest_bindable())
    let parameter_row = @joint_diffusion_cuda_aot.lower_final_adaln_parameter_row_cuda_aot_candidate(
      requirements,
      requirement_ordinal=parameter_row_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(parameter_row.operation(), FinalAdaLnParameterRow(2688, 5376))
    assert_eq(parameter_row.weight_element_count(), 28901376L)
    assert_eq(parameter_row.dimensions().grid_x(), 10752)
    assert_eq(parameter_row.operands()[0].byte_count(), 10752L)
    assert_eq(parameter_row.operands()[1].byte_count(), 57802752L)
    assert_eq(parameter_row.operands()[2].byte_count(), 21504L)
    assert_eq(parameter_row.operands()[3].byte_count(), 10752L)
    assert_eq(parameter_row.operands()[4].byte_count(), 10752L)
    assert_false(parameter_row.manifest_bindable())
    let gather = @joint_diffusion_cuda_aot.lower_packed_modality_adaln_cuda_aot_candidate(
      requirements,
      timestep=timestep_selection,
      modality=modality_selection,
      requirement_ordinal=packed_gather_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(gather.operation(), PackedModalityAdaLnRows(5376))
    assert_eq(gather.packed_row_count(), 37847)
    assert_eq(gather.video_row_count(), 37296)
    assert_eq(gather.audio_row_count(), 414)
    assert_eq(gather.dimensions().grid_x(), 791910)
    assert_eq(gather.operands().length(), 8)
    assert_false(gather.manifest_bindable())
    let frequencies = @joint_diffusion_cuda_aot.lower_timestep_frequency_cuda_aot_candidate(
      requirements,
      values=timestep_values,
      requirement_ordinal=timestep_frequency_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(frequencies.operation(), TimestepFrequencyRows(256))
    assert_eq(frequencies.row_count(), 3)
    assert_eq(frequencies.frequency_width(), 256)
    assert_eq(frequencies.dimensions().grid_x(), 3)
    assert_eq(frequencies.operands()[0].byte_count(), 12L)
    assert_eq(frequencies.operands()[1].byte_count(), 3072L)
    assert_false(frequencies.manifest_bindable())
    assert_timestep_mlp_candidate(requirements, timestep_values, frequencies)
    try
      @joint_diffusion_cuda_aot.lower_timestep_frequency_cuda_aot_candidate(
        requirements,
        values=timestep_values,
        requirement_ordinal=0,
        target=integration_target(),
        compiler=integration_compiler(),
        limits=integration_limits(),
      )
    catch {
      UnsupportedRequirement(0) => ()
      _ => fail("unexpected timestep-frequency ordinal rejection")
    } noraise {
      _ => fail("conditioning ordinal admitted as timestep frequencies")
    }
    let video_output = @joint_diffusion_cuda_aot.lower_output_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(video_output.numerical_contract(), OrderedFloat32DenseF32OutputV1)
    assert_eq(
      video_output.operation(),
      VideoOutputProjection(24, 1, 2, 2, 5376),
    )
    assert_eq(video_output.capability(), VideoOutputProjectionAot)
    assert_eq(video_output.row_count(), 37296L)
    assert_eq(video_output.input_width(), 5376)
    assert_eq(video_output.output_width(), 96)
    assert_eq(video_output.input_element_count(), 200503296L)
    assert_eq(video_output.output_element_count(), 3580416L)
    assert_eq(video_output.projection_weight_element_count(), 516096L)
    assert_eq(video_output.projection_bias_element_count(), 96L)
    assert_eq(video_output.model_weight_element_count(), 516192L)
    assert_eq(video_output.dimensions().grid_x(), 13986)
    assert_eq(video_output.operands()[0].role(), VideoF32HeadInput)
    assert_eq(video_output.operands()[0].byte_count(), 802013184L)
    assert_eq(video_output.operands()[1].role(), ProjectionWeight)
    assert_eq(video_output.operands()[1].byte_count(), 2064384L)
    assert_eq(video_output.operands()[2].role(), ProjectionBias)
    assert_eq(video_output.operands()[2].byte_count(), 384L)
    assert_eq(video_output.operands()[3].role(), VideoPredictionOutput)
    assert_eq(video_output.operands()[3].byte_count(), 14321664L)
    assert_false(video_output.manifest_bindable())
    let audio_output = @joint_diffusion_cuda_aot.lower_output_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=output_projection_ordinal(requirements, AudioLatent),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(audio_output.operation(), AudioOutputProjection(32, 5376))
    assert_eq(audio_output.capability(), AudioOutputProjectionAot)
    assert_eq(audio_output.row_count(), 414L)
    assert_eq(audio_output.input_width(), 5376)
    assert_eq(audio_output.output_width(), 32)
    assert_eq(audio_output.input_element_count(), 2225664L)
    assert_eq(audio_output.output_element_count(), 13248L)
    assert_eq(audio_output.projection_weight_element_count(), 172032L)
    assert_eq(audio_output.projection_bias_element_count(), 32L)
    assert_eq(audio_output.model_weight_element_count(), 172064L)
    assert_eq(audio_output.dimensions().grid_x(), 52)
    assert_eq(audio_output.operands()[0].role(), AudioF32HeadInput)
    assert_eq(audio_output.operands()[0].byte_count(), 8902656L)
    assert_eq(audio_output.operands()[1].byte_count(), 688128L)
    assert_eq(audio_output.operands()[2].byte_count(), 128L)
    assert_eq(audio_output.operands()[3].role(), AudioPredictionOutput)
    assert_eq(audio_output.operands()[3].byte_count(), 52992L)
    assert_false(audio_output.manifest_bindable())
    recipe_digests.push(video.recipe_digest())
    recipe_digests.push(video_norm.recipe_digest())
  }
  assert_eq(recipe_digests.length(), 4)
  assert_true(recipe_digests[0] != recipe_digests[1])
  assert_true(recipe_digests[1] != recipe_digests[3])
}
