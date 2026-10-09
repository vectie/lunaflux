///|
fn validate_token_hash_launch_geometry(
  plan : DeepSeekV4TokenHashOperandPlan,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
) -> Unit raise DeepSeekV4TokenHashLaunchBindingError {
  validate_candidate_for_artifact_binding(plan, candidate) catch {
    error => raise ArtifactBinding(error)
  }
  let maximum_tokens = candidate.maximum_tokens()
  guard candidate.operation()
    is TokenHashRouting(3, 129280, experts_per_token, routed_expert_count) else {
    raise CandidateGeometryMismatch(-1)
  }
  if maximum_tokens <= 0 ||
    experts_per_token != 6 ||
    routed_expert_count != plan.routed_expert_count {
    raise CandidateGeometryMismatch(-1)
  }
  let token_bytes = maximum_tokens.to_int64() * 4L
  let output_bytes = maximum_tokens.to_int64() *
    experts_per_token.to_int64() *
    4L
  let expected : Array[
    (
      @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
      Int64,
      Int64,
    ),
  ] = [
    (StepCountsInput, 20L, 4L),
    (TokenIdsInput, token_bytes, 4L),
    (TokenHashTableInput, 129280L * 6L * 4L, 4L),
    (RoutingIndicesOutput, output_bytes, 4L),
  ]
  let operands = candidate.operands()
  if operands.length() != expected.length() {
    raise CandidateGeometryMismatch(-1)
  }
  for ordinal, entry in expected {
    let (role, bytes, alignment) = entry
    if operands[ordinal].role() != role ||
      operands[ordinal].byte_count() != bytes ||
      operands[ordinal].alignment() != alignment {
      raise CandidateGeometryMismatch(ordinal)
    }
  }
}

///|
fn validate_token_hash_sidecar_region(
  plan : DeepSeekV4TokenHashOperandPlan,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise DeepSeekV4TokenHashLaunchBindingError {
  if layout.source_identity() != plan.source_identity ||
    layout.digest() != plan.sidecar_layout_digest ||
    layout.device_arena_bytes() <= 0L {
    raise SidecarLayoutMismatch
  }
  guard layout.region(plan.region_ordinal) is Some(region) else {
    raise SidecarLayoutMismatch
  }
  if region.tensor_ordinal() != plan.region_tensor_ordinal ||
    region.device_offset() != plan.device_offset ||
    region.byte_count() != plan.byte_count ||
    plan.device_offset > layout.device_arena_bytes() - plan.byte_count {
    raise SidecarLayoutMismatch
  }
}

///|
fn validate_token_hash_upstream_bindings(
  plan : DeepSeekV4TokenHashOperandPlan,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
  artifact : DeepSeekV4TokenHashArtifactBinding,
  compile : DeepSeekV4TokenHashCompileEvidenceBinding,
) -> Unit raise DeepSeekV4TokenHashLaunchBindingError {
  if artifact.status() != InertUnqualified ||
    artifact.operand_plan_digest() != plan.digest ||
    artifact.source_identity() != plan.source_identity ||
    artifact.requirements_digest() != plan.requirements_digest ||
    artifact.layer_index() != plan.layer_index ||
    artifact.requirement_ordinal() != plan.requirement_ordinal ||
    artifact.target() != candidate.target() ||
    artifact.function_symbol() != candidate.function_symbol() ||
    artifact.dimensions() != candidate.dimensions() ||
    artifact.operand_count() != 4 ||
    artifact.token_hash_operand_ordinal() != 2 ||
    artifact.token_hash_device_offset() != plan.device_offset {
    raise ArtifactEvidenceMismatch
  }
  if compile.status() != InertSelfConsistentUnqualified ||
    compile.artifact_binding_digest() != artifact.digest() ||
    compile.source_identity() != plan.source_identity ||
    compile.requirements_digest() != plan.requirements_digest ||
    compile.layer_index() != plan.layer_index ||
    compile.requirement_ordinal() != plan.requirement_ordinal ||
    compile.source_digest_sha256() != candidate.source_digest().as_hex() ||
    compile.recipe_digest_sha256() != candidate.recipe_digest().as_hex() ||
    compile.target() != candidate.target() ||
    compile.function_symbol() != candidate.function_symbol() ||
    compile.module_digest() != artifact.module_digest() ||
    compile.module_byte_count() != artifact.module_byte_count() {
    raise CompileEvidenceMismatch
  }
}

///|
/// Completes inert allocation-relative metadata for all four token-hash
/// operands. It preserves operand 2 from the authenticated sidecar layout and
/// accepts bounded descriptors for operands 0, 1, and 3. No pointer, loader,
/// stream, live-device, or launch authority enters the result.
pub fn bind_complete_token_hash_launch(
  plan : DeepSeekV4TokenHashOperandPlan,
  sidecar_layout : @segmented_device_materialize.SegmentedDeviceLayout,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
  artifact : DeepSeekV4TokenHashArtifactBinding,
  compile : DeepSeekV4TokenHashCompileEvidenceBinding,
  step_counts_region : @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionInput,
  token_ids_region : @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionInput,
  routing_indices_region : @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionInput,
  region_limits : @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionPlanLimits,
) -> DeepSeekV4TokenHashLaunchBinding raise DeepSeekV4TokenHashLaunchBindingError {
  validate_token_hash_launch_geometry(plan, candidate)
  validate_token_hash_sidecar_region(plan, sidecar_layout)
  validate_token_hash_upstream_bindings(plan, candidate, artifact, compile)
  let table_region = @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionInput::new(
    operand_ordinal=2,
    role=TokenHashTableInput,
    allocation_identity_sha256=sidecar_layout.digest().as_hex(),
    device_offset=plan.device_offset,
    allocation_capacity_bytes=sidecar_layout.device_arena_bytes(),
  ) catch {
    error => raise OperandRegions(error)
  }
  let operand_regions = @advanced_decoder_operand_region_plan.plan_advanced_decoder_operand_regions(
    candidate.operands(),
    [step_counts_region, token_ids_region, table_region, routing_indices_region],
    region_limits,
  ) catch {
    error => raise OperandRegions(error)
  }
  let without_digest : DeepSeekV4TokenHashLaunchBinding = {
    source_identity: plan.source_identity,
    requirements_digest: plan.requirements_digest,
    layer_index: plan.layer_index,
    requirement_ordinal: plan.requirement_ordinal,
    maximum_tokens: candidate.maximum_tokens(),
    experts_per_token: 6,
    routed_expert_count: plan.routed_expert_count,
    operand_plan_digest: plan.digest,
    sidecar_layout_digest: sidecar_layout.digest(),
    artifact_binding_digest: artifact.digest(),
    compile_binding_digest: compile.digest(),
    compile_evidence_digest: compile.compile_evidence_digest(),
    operand_regions,
    source_digest_sha256: candidate.source_digest().as_hex(),
    recipe_digest_sha256: candidate.recipe_digest().as_hex(),
    compiler: candidate.compiler_policy(),
    target: candidate.target(),
    entry_point: artifact.entry_point(),
    function_symbol: candidate.function_symbol(),
    module_digest: artifact.module_digest(),
    dimensions: candidate.dimensions(),
    digest: DeepSeekV4TokenHashLaunchBindingDigest(String::make(64, '0')),
  }
  {
    ..without_digest,
    digest: canonical_token_hash_launch_binding_digest(without_digest),
  }
}
