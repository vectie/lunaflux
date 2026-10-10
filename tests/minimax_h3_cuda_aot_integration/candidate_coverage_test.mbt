///|
fn minimax_coverage_limits() -> @aot_candidate_coverage.AotCandidateCoverageLimits {
  try! @aot_candidate_coverage.AotCandidateCoverageLimits::new(
    maximum_requirements=64,
    maximum_candidates=64,
    maximum_encoding_bytes=65536,
  )
}

///|
fn minimax_candidate_evidence(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  io : @joint_diffusion_io_plan.JointDiffusionIoPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  timestep : @joint_diffusion_io_plan.JointPackedTimestepSelection,
  modality : @joint_diffusion_io_plan.JointPackedModalitySelection,
  values : @joint_diffusion_io_plan.JointDistinctTimestepValues,
) -> Array[@aot_candidate_coverage.AotCandidateEvidenceInput] {
  let evidence : Array[@aot_candidate_coverage.AotCandidateEvidenceInput] = []
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      JointTransformerAttention(_, _, _, _, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
          requirements,
          request=io,
          packed_timestep=timestep,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_transformer_attention(
            candidate,
          ),
        )
      }
      JointTransformerMlp(_, _, _, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_transformer_mlp_cuda_aot_candidate(
          requirements,
          request=io,
          packed_timestep=timestep,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_transformer_mlp(
            candidate,
          ),
        )
      }
      VideoLatentProjection(_, _, _, _, _) | AudioLatentProjection(_, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_latent_projection_cuda_aot_candidate(
          requirements,
          request_shape=request,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_latent_projection(
            candidate,
          ),
        )
      }
      VideoFinalRmsNorm(_, _, _) | AudioFinalRmsNorm(_, _, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_final_rms_norm_cuda_aot_candidate(
          requirements,
          request_shape=request,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_final_rms_norm(
            candidate,
          ),
        )
      }
      VideoFinalAdaLnF32Boundary(_) | AudioFinalAdaLnF32Boundary(_) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_final_adaln_f32_cuda_aot_candidate(
          requirements,
          request_shape=request,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_final_adaln(candidate),
        )
      }
      FinalAdaLnParameterRow(_, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_final_adaln_parameter_row_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_final_adaln_parameter_row(
            candidate,
          ),
        )
      }
      PackedModalityAdaLnRows(_) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_packed_modality_adaln_cuda_aot_candidate(
          requirements,
          timestep~,
          modality~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_packed_modality_adaln(
            candidate,
          ),
        )
      }
      TimestepFrequencyRows(_) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_timestep_frequency_cuda_aot_candidate(
          requirements,
          values~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_timestep_frequency(
            candidate,
          ),
        )
      }
      TimestepEmbeddingMlpRows(_, _, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
          requirements,
          values~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_timestep_mlp(candidate),
        )
      }
      DistinctTimestepAdaLnParameterRows(_, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_distinct_timestep_adaln_cuda_aot_candidate(
          requirements,
          values~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_distinct_timestep_adaln(
            candidate,
          ),
        )
      }
      VideoOutputProjection(_, _, _, _, _) | AudioOutputProjection(_, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_output_projection_cuda_aot_candidate(
          requirements,
          request_shape=request,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_output_projection(
            candidate,
          ),
        )
      }
      ShiftedFlowStep(_, _, _, _) => {
        let candidate = try! @joint_diffusion_cuda_aot.lower_shifted_flow_cuda_aot_candidate(
          requirements,
          request_shape=request,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_shifted_flow(candidate),
        )
      }
      VideoVaeLatentDestandardization(_, _, _) => {
        let parameters = official_video_vae_destandardization(variant, request)
        let candidate = try! @joint_diffusion_cuda_aot.lower_video_vae_latent_destandardization_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_video_vae_latent_destandardization(
            candidate,
          ),
        )
      }
      AudioVaeLatentDestandardization(_, _, _, _) => {
        let parameters = official_audio_vae_destandardization(variant, request)
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_latent_destandardization_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_latent_destandardization(
            candidate,
          ),
        )
      }
      AudioVaeDecoderInputProjection(_, _, _, _, _, _) => {
        let parameters = official_audio_vae_decoder_input_projection(
          variant, request,
        )
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_input_projection(
            candidate,
          ),
        )
      }
      AudioVaeDecoderConvPre(_, _, _, _, _, _, _, _, _) => {
        let parameters = official_audio_vae_decoder_conv_pre(variant, request)
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_conv_pre(
            candidate,
          ),
        )
      }
      AudioVaeDecoderFirstUpsample(_, _, _, _, _, _, _, _, _) => {
        let parameters = official_audio_vae_decoder_first_upsample(
          variant, request,
        )
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_upsample_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_first_upsample(
            candidate,
          ),
        )
      }
      AudioVaeDecoderFirstAmpActivation(_, _, _, _, _, _) => {
        let parameters = official_audio_vae_decoder_first_amp_activation(
          variant, request,
        )
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_activation_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_first_amp_activation(
            candidate,
          ),
        )
      }
      AudioVaeDecoderFirstAmpStage(_, _, _, _, _, _) => {
        let parameters = try! @minimax_h3_vae_reference.bind_first_amp_parameter_layout()
        let workspace = try! @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
          element_count=1059840L,
          intermediate_element_count=2119680L,
          normalized_weight_element_count=2883584L,
        )
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_stage_cuda_aot_candidate(
          requirements,
          request~,
          parameter_layout=parameters,
          workspace_layout=workspace,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_first_amp_stage(
            candidate,
          ),
        )
      }
      AudioVaeDecoderUpsampleTransition(stage, _, _, _, _, _, _, _, _, _) => {
        let parameters = if stage == 1 {
          official_audio_vae_decoder_second_upsample(variant, request)
        } else {
          official_audio_vae_decoder_third_upsample(variant, request)
        }
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_upsample_transition(
            candidate,
          ),
        )
      }
      AudioVaeDecoderAmpActivation(stage, _, _, _, _, _, _) => {
        let parameters = if stage == 1 {
          official_audio_vae_decoder_second_amp_activation(variant, request)
        } else {
          official_audio_vae_decoder_third_amp_activation(variant, request)
        }
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_activation_cuda_aot_candidate(
          requirements,
          parameters~,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_amp_activation(
            candidate,
          ),
        )
      }
      AudioVaeDecoderAmpStage(_, _, _, _, _, _, _, _) => {
        let parameters = try! @minimax_h3_vae_reference.bind_second_amp_parameter_layout()
        let workspace = try! @joint_audio_amp_workspace_layout.amp_workspace_layout(
          element_count=2649600L,
          intermediate_element_count=5299200L,
          normalized_weight_element_count=720896L,
        )
        let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_stage_cuda_aot_candidate(
          requirements,
          request~,
          parameter_layout=parameters,
          workspace_layout=workspace,
          requirement_ordinal=ordinal,
          target=integration_target(),
          compiler=integration_compiler(),
          limits=integration_limits(),
        )
        evidence.push(
          try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_amp_stage(
            candidate,
          ),
        )
      }
      _ => ()
    }
  }
  evidence
}

///|
test "official MiniMax candidate coverage remains exact incomplete and inert" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  for profile_index, variant in variants {
    let requirements = official_requirements(variant)
    let (io, timestep, modality, values) = packed_selection_fixture_with_io(
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
    let candidates = minimax_candidate_evidence(
      variant, requirements, io, request, timestep, modality, values,
    )
    let coverage = @aot_candidate_coverage.audit(
      source_identity=requirements.source_identity(),
      requirements_digest_sha256=requirements.digest().as_hex(),
      requirement_count=requirements.requirements().length(),
      candidates~,
      limits=minimax_coverage_limits(),
    )
    assert_eq(coverage.status(), Incomplete)
    assert_eq(coverage.candidate_count(), 29)
    assert_eq(coverage.non_bindable_ordinals().length(), 29)
    assert_eq(coverage.missing_ordinals().length(), 4)
    let conditioning_requirement = requirements.requirements()[coverage.missing_ordinals()[1]]
    assert_eq(
      conditioning_requirement.operation(),
      if profile_index == 0 {
        OrderedKeyframeConditioning(2, 24)
      } else {
        OrderedReferenceConditioning(9, 3, 3, 12, false)
      },
    )
    try coverage.require_complete() catch {
      IncompleteCoverage(4) => ()
      _ => fail("MiniMax candidate coverage returned the wrong gap")
    } noraise {
      _ => fail("partial MiniMax candidate coverage was promoted")
    }
    if profile_index == 0 {
      assert_eq(
        coverage.digest().as_hex(),
        "ae07b9c1866f6506098c08ee73ca25e8d4c67bc2298c998c8efe9a14e6796a78",
      )
      continue
    }
    inspect(
      coverage.digest().as_hex(),
      content=(
        #|257778149a3a4c8d39d66a17048fe88268ad59d2600ad56ece19149211167d3f
      ),
    )
  }
}
