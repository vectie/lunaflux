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
pub fn from_foundation(
  candidate : @glm_hybrid_cuda_aot.GlmHybridFoundationCudaCandidate,
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
pub fn from_dense_projection(
  candidate : @glm_hybrid_cuda_aot.GlmHybridDenseProjectionCudaCandidate,
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
pub fn from_dsa_pooling(
  candidate : @glm_hybrid_cuda_aot.GlmHybridDsaPoolingCudaCandidate,
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
pub fn from_kv_assembly(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKvAssemblyCudaCandidate,
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
pub fn from_dsa_index(
  candidate : @glm_hybrid_cuda_aot.GlmHybridDsaIndexCudaCandidate,
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
pub fn from_sparse_attention(
  candidate : @glm_hybrid_cuda_aot.GlmHybridSparseAttentionCudaCandidate,
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
pub fn from_kv_split_norm(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKvSplitNormCudaCandidate,
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
pub fn from_rotary(
  candidate : @glm_hybrid_cuda_aot.GlmHybridRotaryCudaCandidate,
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
pub fn from_dense_swiglu(
  candidate : @glm_hybrid_cuda_aot.GlmHybridDenseSwiGluCudaCandidate,
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
pub fn from_kda_short_convolution(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKdaShortConvolutionCudaCandidate,
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
pub fn from_kda_recurrent(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKdaRecurrentCudaCandidate,
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
pub fn from_kda_projection(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKdaProjectionCudaCandidate,
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
pub fn from_kda_output_gate(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKdaOutputGateCudaCandidate,
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
pub fn from_kda_gated_norm(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKdaGatedNormCudaCandidate,
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
pub fn from_kda_decay_control(
  candidate : @glm_hybrid_cuda_aot.GlmHybridKdaDecayControlCudaCandidate,
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
pub fn from_moe_router(
  candidate : @glm_hybrid_cuda_aot.GlmHybridMoeRouterCudaCandidate,
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
pub fn from_moe_sigmoid_correction(
  candidate : @glm_hybrid_cuda_aot.GlmHybridMoeScoreCudaCandidate,
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
pub fn from_moe_grouped_routing(
  candidate : @glm_hybrid_cuda_aot.GlmHybridMoeGroupedRoutingCudaCandidate,
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
pub fn from_moe_feed_forward(
  candidate : @glm_hybrid_cuda_aot.GlmHybridMoeFeedForwardCudaCandidate,
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
pub fn from_hyper_control(
  candidate : @glm_hybrid_cuda_aot.GlmHybridHyperControlCudaCandidate,
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
