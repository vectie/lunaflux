///|
fn evidence(
  source_identity : @spec.ModelIdentity,
  requirements_digest_sha256 : String,
  requirement_ordinal : Int,
  source_digest_sha256 : String,
  recipe_digest_sha256 : String,
  manifest_bindable : Bool,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity~,
    requirements_digest_sha256~,
    requirement_ordinal~,
    source_digest_sha256~,
    recipe_digest_sha256~,
    manifest_bindable~,
  )
}

///|
pub fn from_latent_projection(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionLatentProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_output_projection(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionOutputProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_final_rms_norm(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionFinalRmsNormCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_final_adaln_parameter_row(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionFinalAdaLnParameterRowCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_final_adaln(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionFinalAdaLnF32CudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_shifted_flow(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionShiftedFlowCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_packed_modality_adaln(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionPackedModalityAdaLnCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_timestep_frequency(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionTimestepFrequencyCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_timestep_mlp(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionTimestepMlpCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_distinct_timestep_adaln(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionDistinctTimestepAdaLnCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_transformer_mlp(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionTransformerMlpCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_transformer_attention(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionTransformerAttentionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_video_vae_latent_destandardization(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionVideoVaeLatentDestandardizationCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_latent_destandardization(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeLatentDestandardizationCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_input_projection(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_conv_pre(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeConvPreCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_first_upsample(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeFirstUpsampleCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_first_amp_activation(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeFirstAmpCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_first_amp_stage(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeFirstAmpStageCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_upsample_transition(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeUpsampleTransitionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_amp_activation(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeAmpActivationCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}

///|
pub fn from_audio_vae_decoder_amp_stage(
  candidate : @joint_diffusion_cuda_aot.JointDiffusionAudioVaeAmpStageCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  evidence(
    candidate.source_identity(),
    candidate.requirements_digest().as_hex(),
    candidate.requirement_ordinal(),
    candidate.source_digest().as_hex(),
    candidate.recipe_digest().as_hex(),
    candidate.manifest_bindable(),
  )
}
