///|
fn launch_operand_equal(
  left : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  right : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
) -> Bool {
  left.role() == right.role() &&
  left.byte_count() == right.byte_count() &&
  left.alignment() == right.alignment()
}

///|
fn candidate_operands_match_plan(
  plan : DeepSeekV4TokenHashOperandPlan,
  operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
) -> Bool {
  operands.length() == 4 &&
  operands[plan.operand_ordinal].role() == plan.operand_role &&
  operands[plan.operand_ordinal].byte_count() == plan.byte_count &&
  operands[plan.operand_ordinal].alignment() == plan.alignment
}

///|
fn validate_candidate_for_artifact_binding(
  plan : DeepSeekV4TokenHashOperandPlan,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
) -> Unit raise DeepSeekV4TokenHashArtifactBindingError {
  if candidate.source_identity() != plan.source_identity {
    raise CandidateIdentityMismatch
  }
  if candidate.requirements_digest() != plan.requirements_digest {
    raise CandidateRequirementsDigestMismatch
  }
  if candidate.requirement_ordinal() != plan.requirement_ordinal {
    raise CandidateOrdinalMismatch
  }
  guard candidate.operation()
    is TokenHashRouting(3, 129280, 6, routed_expert_count) else {
    raise CandidateOperationMismatch
  }
  if routed_expert_count != plan.routed_expert_count {
    raise CandidateOperationMismatch
  }
  if candidate.capability() != TokenHashRoutingAot {
    raise CandidateCapabilityMismatch
  }
  if candidate.family() != TokenHashRoutingI32 ||
    candidate.numerical_contract() != BitExactInt32TableLookupV1 {
    raise CandidateContractMismatch
  }
  if candidate.source_digest() != plan.candidate_source_digest {
    raise CandidateSourceDigestMismatch
  }
  if candidate.recipe_digest() != plan.candidate_recipe_digest {
    raise CandidateRecipeDigestMismatch
  }
  if candidate.manifest_bindable() {
    raise CandidateUnexpectedlyBindable
  }
  let operands = candidate.operands()
  if !candidate_operands_match_plan(plan, operands) {
    raise CandidateOperandMismatch(plan.operand_ordinal)
  }
}

///|
fn find_artifact_entry_point(
  admission : @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmission,
  identity : @catalog.AotKernelEntryPoint,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderAdmittedEntryPoint? {
  for entry in admission.entry_points() {
    if entry.identity() == identity {
      return Some(entry)
    }
  }
  None
}

///|
fn validate_launch_operands(
  candidate : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  admitted : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
) -> Unit raise DeepSeekV4TokenHashArtifactBindingError {
  if candidate.length() != admitted.length() {
    raise ArtifactOperandMismatch(-1)
  }
  for ordinal, operand in candidate {
    if !launch_operand_equal(operand, admitted[ordinal]) {
      raise ArtifactOperandMismatch(ordinal)
    }
  }
}

///|
/// Joins one exact token-hash operand plan to digest-verified module, symbol,
/// and launch evidence. The result remains inert and grants no loader or launch
/// authority.
pub fn bind_token_hash_artifact(
  plan : DeepSeekV4TokenHashOperandPlan,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
  admission : @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmission,
) -> DeepSeekV4TokenHashArtifactBinding raise DeepSeekV4TokenHashArtifactBindingError {
  validate_candidate_for_artifact_binding(plan, candidate)
  if admission.source_identity() != plan.source_identity {
    raise ArtifactIdentityMismatch
  }
  if admission.requirements_digest() != plan.requirements_digest {
    raise ArtifactRequirementsDigestMismatch
  }
  if admission.target() != candidate.target() {
    raise ArtifactTargetMismatch
  }
  if admission.status() != InertUnqualified {
    raise ArtifactUnexpectedStatus
  }
  guard admission.launches().get(plan.requirement_ordinal) is Some(launch) else {
    raise ArtifactLaunchMissing(plan.requirement_ordinal)
  }
  if launch.requirement_ordinal() != plan.requirement_ordinal ||
    launch.operation() != candidate.operation() ||
    launch.capability() != candidate.capability() ||
    launch.dimensions() != candidate.dimensions() {
    raise ArtifactLaunchMismatch
  }
  validate_launch_operands(candidate.operands(), launch.operands())
  guard find_artifact_entry_point(admission, launch.entry_point())
    is Some(entry) else {
    raise ArtifactEntryPointMissing
  }
  if entry.function_symbol() != candidate.function_symbol() {
    raise ArtifactSymbolMismatch
  }
  guard admission.modules().get(entry.module_index()) is Some(module_) else {
    raise ArtifactModuleMissing
  }
  if module_.digest() != launch.module_digest() ||
    module_.digest() != launch.entry_point().family().module_digest() ||
    module_.module_bytes().is_empty() {
    raise ArtifactModuleMismatch
  }
  let without_digest : DeepSeekV4TokenHashArtifactBinding = {
    source_identity: plan.source_identity,
    requirements_digest: plan.requirements_digest,
    layer_index: plan.layer_index,
    requirement_ordinal: plan.requirement_ordinal,
    operand_plan_digest: plan.digest,
    candidate_source_digest: plan.candidate_source_digest,
    candidate_recipe_digest: plan.candidate_recipe_digest,
    artifact_admission_digest: admission.digest(),
    target: admission.target(),
    module_digest: module_.digest(),
    module_byte_count: module_.module_bytes().length(),
    entry_point: entry.identity(),
    function_symbol: entry.function_symbol(),
    dimensions: launch.dimensions(),
    operand_count: launch.operands().length(),
    token_hash_operand_ordinal: plan.operand_ordinal,
    token_hash_device_offset: plan.device_offset,
    digest: DeepSeekV4TokenHashArtifactBindingDigest(String::make(64, '0')),
  }
  {
    ..without_digest,
    digest: canonical_artifact_binding_digest(without_digest),
  }
}
