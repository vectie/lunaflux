///|
fn fused_manifest_require_compiled(
  compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  receipt : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> Unit raise FusedPhysicalEvidenceError {
  if compiled.manifest_bindable() ||
    compiled.family() != family ||
    compiled.candidate_digest() != candidate ||
    compiled.source_digest() != source ||
    compiled.recipe_digest() != recipe ||
    compiled.module_digest() != module_digest ||
    compiled.compile_receipt_digest() != receipt {
    raise FusedPhysicalEvidenceMismatch(Artifact)
  }
}

///|
fn fused_manifest_require_benchmark(
  evidence : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> Unit raise FusedPhysicalEvidenceError {
  if evidence.manifest_bindable() ||
    evidence.promotion_authority() ||
    evidence.family() != family ||
    evidence.candidate_digest() != candidate ||
    evidence.shape_set_identity() !=
    @luna_fused_candidate_qualification.fused_candidate_benchmark_shape_set_identity() ||
    evidence.observation_count() <= 0 {
    raise FusedPhysicalEvidenceMismatch(Benchmark)
  }
}

///|
fn fused_manifest_approval_canonical(
  physical : AdmittedFusedPhysicalEvidence,
  qkv : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  residual : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  qkv_benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
  residual_benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-fused-manifest-approval-record.v1\n" +
    "scope=exact-sm120-fused-pair\n" +
    "physical_evidence_sha256=" +
    physical.digest().as_hex() +
    "\n" +
    "artifact_directory_seal_sha256=" +
    physical.artifact_directory_seal_digest().as_hex() +
    "\nevidence_directory_seal_sha256=" +
    physical.evidence_directory_seal_digest().as_hex() +
    "\nqkv_candidate_sha256=" +
    qkv.candidate_digest().as_hex() +
    "\nqkv_source_sha256=" +
    qkv.source_digest().as_hex() +
    "\nqkv_recipe_sha256=" +
    qkv.recipe_digest().as_hex() +
    "\nqkv_cubin_sha256=" +
    qkv.module_digest().as_hex() +
    "\nqkv_compile_receipt_sha256=" +
    qkv.compile_receipt_digest().as_hex() +
    "\nqkv_compiled_binding_sha256=" +
    qkv.binding_digest().as_hex() +
    "\nqkv_benchmark_sha256=" +
    qkv_benchmark.digest().as_hex() +
    "\nresidual_candidate_sha256=" +
    residual.candidate_digest().as_hex() +
    "\nresidual_source_sha256=" +
    residual.source_digest().as_hex() +
    "\nresidual_recipe_sha256=" +
    residual.recipe_digest().as_hex() +
    "\nresidual_cubin_sha256=" +
    residual.module_digest().as_hex() +
    "\nresidual_compile_receipt_sha256=" +
    residual.compile_receipt_digest().as_hex() +
    "\nresidual_compiled_binding_sha256=" +
    residual.binding_digest().as_hex() +
    "\nresidual_benchmark_sha256=" +
    residual_benchmark.digest().as_hex() +
    "\nbenchmark_shape_set_sha256=" +
    qkv_benchmark.shape_set_identity().as_hex() +
    "\nphysical_correctness_required=true\n" +
    "paired_benchmark_required=true\n" +
    "external_signature_required=true\n" +
    "requested_authority=manifest-bound-fused-pair\n",
  )
}

///|
/// Constructs the exact record that an external deployment authority may sign.
/// This step grants no manifest or promotion authority.
pub fn prepare_fused_manifest_approval_record(
  physical : AdmittedFusedPhysicalEvidence,
  qkv_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  residual_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  qkv_benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
  residual_benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence,
) -> FusedManifestApprovalRecord raise FusedPhysicalEvidenceError {
  if physical.manifest_bindable() || physical.promotion_authorized() {
    raise FusedPhysicalEvidenceMismatch(Authority)
  }
  fused_manifest_require_compiled(
    qkv_compiled,
    QkvPositionedRopePagedKvWrite,
    physical.qkv_candidate_digest(),
    physical.qkv_source_digest(),
    physical.qkv_recipe_digest(),
    physical.qkv_cubin_digest(),
    physical.qkv_compile_receipt_digest(),
  )
  fused_manifest_require_compiled(
    residual_compiled,
    ResidualRmsNorm,
    physical.residual_candidate_digest(),
    physical.residual_source_digest(),
    physical.residual_recipe_digest(),
    physical.residual_cubin_digest(),
    physical.residual_compile_receipt_digest(),
  )
  fused_manifest_require_benchmark(
    qkv_benchmark,
    QkvPositionedRopePagedKvWrite,
    physical.qkv_candidate_digest(),
  )
  fused_manifest_require_benchmark(
    residual_benchmark,
    ResidualRmsNorm,
    physical.residual_candidate_digest(),
  )
  if qkv_benchmark.shape_set_identity() !=
    residual_benchmark.shape_set_identity() {
    raise FusedPhysicalEvidenceMismatch(Benchmark)
  }
  let canonical = fused_manifest_approval_canonical(
    physical, qkv_compiled, residual_compiled, qkv_benchmark, residual_benchmark,
  )
  if canonical.length() > FUSED_PHYSICAL_MAXIMUM_CANONICAL_BYTES {
    raise FusedPhysicalEvidenceLimitExceeded
  }
  {
    physical,
    qkv_compiled,
    residual_compiled,
    qkv_benchmark,
    residual_benchmark,
    canonical,
    digest: fused_physical_digest(canonical),
  }
}

///|
/// Converts only an exact deployment-approved record into fused manifest
/// authority. Only an opaque, authenticated external approval can cross this
/// deployment-owned trust boundary for the exact record. No current production
/// manifest loader, worker, service, or runtime route consumes the wrapper.
pub fn admit_external_fused_manifest_approval(
  record : FusedManifestApprovalRecord,
  envelope : @luna_capability_manifest.LunaDetachedSignatureEnvelope,
  approval : @luna_capability_manifest.LunaAuthenticatedExternalApproval,
) -> ApprovedFusedManifestArtifacts raise FusedPhysicalEvidenceError {
  let admitted = @luna_capability_manifest.admit_luna_external_approved_record(
    record.canonical,
    envelope,
    approval,
    maximum_canonical_bytes=FUSED_PHYSICAL_MAXIMUM_CANONICAL_BYTES,
  ) catch {
    _ => raise FusedPhysicalEvidenceMismatch(ExternalApproval)
  }
  if admitted.record_digest().as_hex() != record.digest.as_hex() {
    raise FusedPhysicalEvidenceMismatch(ExternalApproval)
  }
  { record, approval: admitted }
}
