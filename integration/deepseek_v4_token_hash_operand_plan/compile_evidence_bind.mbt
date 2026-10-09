///|
fn validate_compile_evidence_candidate(
  binding : DeepSeekV4TokenHashArtifactBinding,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
) -> Unit raise DeepSeekV4TokenHashCompileEvidenceBindingError {
  if candidate.source_identity() != binding.source_identity {
    raise CandidateIdentityMismatch
  }
  if candidate.requirements_digest() != binding.requirements_digest {
    raise CandidateRequirementsDigestMismatch
  }
  if candidate.requirement_ordinal() != binding.requirement_ordinal {
    raise CandidateOrdinalMismatch
  }
  guard candidate.operation()
    is TokenHashRouting(3, 129280, 6, routed_expert_count) else {
    raise CandidateOperationMismatch
  }
  if routed_expert_count <= 0 {
    raise CandidateOperationMismatch
  }
  if candidate.capability() != TokenHashRoutingAot {
    raise CandidateCapabilityMismatch
  }
  if candidate.family() != TokenHashRoutingI32 ||
    candidate.numerical_contract() != BitExactInt32TableLookupV1 {
    raise CandidateContractMismatch
  }
  if candidate.source_digest() != binding.candidate_source_digest {
    raise CandidateSourceDigestMismatch
  }
  if candidate.recipe_digest() != binding.candidate_recipe_digest {
    raise CandidateRecipeDigestMismatch
  }
  if candidate.manifest_bindable() {
    raise CandidateUnexpectedlyBindable
  }
}

///|
/// Joins exact token-hash candidate and artifact identities to an independently
/// admitted, deterministic offline-compile receipt. This authenticates only
/// receipt/module self-consistency; it does not authenticate the producer or
/// grant compiler, loader, launch, or physical-qualification authority.
pub fn bind_token_hash_compile_evidence(
  binding : DeepSeekV4TokenHashArtifactBinding,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
  evidence : @cuda_offline_compile_evidence.CudaOfflineCompileEvidence,
) -> DeepSeekV4TokenHashCompileEvidenceBinding raise DeepSeekV4TokenHashCompileEvidenceBindingError {
  validate_compile_evidence_candidate(binding, candidate)
  if evidence.status() != InertSelfConsistent {
    raise EvidenceUnexpectedStatus
  }
  if evidence.source_digest_sha256() != candidate.source_digest().as_hex() {
    raise EvidenceSourceDigestMismatch
  }
  if evidence.recipe_digest_sha256() != candidate.recipe_digest().as_hex() {
    raise EvidenceRecipeDigestMismatch
  }
  if evidence.compiler_policy() != candidate.compiler_policy() {
    raise EvidenceCompilerMismatch
  }
  if evidence.target() != candidate.target() ||
    evidence.target() != binding.target {
    raise EvidenceTargetMismatch
  }
  if evidence.function_symbol() != candidate.function_symbol() ||
    evidence.function_symbol() != binding.function_symbol {
    raise EvidenceSymbolMismatch
  }
  if evidence.module_digest() != binding.module_digest {
    raise EvidenceModuleDigestMismatch
  }
  if evidence.module_byte_count() != binding.module_byte_count {
    raise EvidenceModuleByteCountMismatch
  }
  let without_digest : DeepSeekV4TokenHashCompileEvidenceBinding = {
    source_identity: binding.source_identity,
    requirements_digest: binding.requirements_digest,
    layer_index: binding.layer_index,
    requirement_ordinal: binding.requirement_ordinal,
    artifact_binding_digest: binding.digest,
    compile_evidence_digest: evidence.digest(),
    source_digest_sha256: evidence.source_digest_sha256(),
    recipe_digest_sha256: evidence.recipe_digest_sha256(),
    driver_identity_sha256: evidence.driver_identity_sha256(),
    target: evidence.target(),
    function_symbol: evidence.function_symbol(),
    module_digest: evidence.module_digest(),
    module_byte_count: evidence.module_byte_count(),
    digest: DeepSeekV4TokenHashCompileEvidenceBindingDigest(
      String::make(64, '0'),
    ),
  }
  {
    ..without_digest,
    digest: canonical_compile_evidence_binding_digest(without_digest),
  }
}
