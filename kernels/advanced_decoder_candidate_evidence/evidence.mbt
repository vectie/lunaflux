///|
pub fn from_foundation(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderFoundationCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_projection(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_key_value_projection(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderKeyValueProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_routing(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_compressed_index(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderCompressedIndexCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_window_compressed_join(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderWindowCompressedJoinCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_compressed_kv_prepare(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderCompressedKvPrepareCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_sparse_attention(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderSparseAttentionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_rotary(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRotaryCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_inverse_rotary(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderInverseRotaryCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_output_a_projection(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderOutputAProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_output_b_projection(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderOutputBProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_query_a_projection(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderQueryAProjectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_hyper_connection(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderHyperConnectionCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_hyper_connection_pre_reduce(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderHyperConnectionPreReduceCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_hyper_connection_post_combine(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderHyperConnectionPostCombineCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}

///|
pub fn from_hyper_connection_head_reduce(
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderHyperConnectionHeadReduceCudaCandidate,
) -> @aot_candidate_coverage.AotCandidateEvidenceInput raise @aot_candidate_coverage.AotCandidateCoverageError {
  @aot_candidate_coverage.AotCandidateEvidenceInput::new(
    source_identity=candidate.source_identity(),
    requirements_digest_sha256=candidate.requirements_digest().as_hex(),
    requirement_ordinal=candidate.requirement_ordinal(),
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    manifest_bindable=candidate.manifest_bindable(),
  )
}
