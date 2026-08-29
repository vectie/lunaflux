///|
fn fused_artifact_purpose_name(
  purpose : @luna_cuda_fused_parallel_aot.FusedParallelArtifactPurpose,
) -> String {
  match purpose {
    QualificationDiagnostics => "qualification-diagnostics"
    ProductionFastPath => "production-fast-path"
  }
}

///|
fn fused_catalog_canonical(
  manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  target : @catalog.DeviceTarget,
  qkv : ApprovedFusedKernelModule,
  residual : ApprovedFusedKernelModule,
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-approved-fused-kernel-catalog.v1\n" +
    "manifest_sha256=" +
    manifest.as_hex() +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nqkv_candidate_sha256=" +
    qkv.candidate.as_hex() +
    "\nqkv_compiled_binding_sha256=" +
    qkv.binding_digest.as_hex() +
    "\nqkv_cubin_sha256=" +
    qkv.module_digest.as_hex() +
    "\nqkv_dispatch_canary_per_token=" +
    qkv.dispatch_canary_per_token.to_string() +
    "\nqkv_artifact_purpose=" +
    fused_artifact_purpose_name(qkv.purpose) +
    "\nresidual_candidate_sha256=" +
    residual.candidate.as_hex() +
    "\nresidual_compiled_binding_sha256=" +
    residual.binding_digest.as_hex() +
    "\nresidual_cubin_sha256=" +
    residual.module_digest.as_hex() +
    "\nresidual_dispatch_canary_per_token=" +
    residual.dispatch_canary_per_token.to_string() +
    "\nresidual_artifact_purpose=" +
    fused_artifact_purpose_name(residual.purpose) +
    "\nentry_count=2\nselection=exact-family-target-manifest-candidate-binding\n",
  )
}

///|
/// Admits an exact two-entry catalog directly from the opaque production
/// manifest. This deliberately does not accept a caller-assembled bundle.
pub fn admit_approved_fused_kernel_catalog(
  manifest_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  approval_record_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  target : @catalog.DeviceTarget,
  qkv_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  residual_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  qkv_benchmark_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  residual_benchmark_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  qkv_dispatch_canary_per_token : UInt,
  residual_dispatch_canary_per_token : UInt,
  qkv_fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity],
  residual_fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity],
  approval : @luna_capability_manifest.LunaAdmittedExternalRecord,
) -> ApprovedFusedKernelCatalog raise FusedArtifactAdmissionError {
  let (qkv, residual) = fused_admit_pair(
    manifest_digest, approval_record_digest, target, qkv_compiled, residual_compiled,
    qkv_benchmark_digest, residual_benchmark_digest, qkv_dispatch_canary_per_token,
    residual_dispatch_canary_per_token, qkv_fallbacks, residual_fallbacks, approval,
  )
  let canonical = fused_catalog_canonical(
    manifest_digest, target, qkv, residual,
  )
  {
    manifest: manifest_digest,
    target,
    qkv,
    residual,
    canonical,
    digest: fused_artifact_digest(canonical),
  }
}

///|
/// Resolves only an exact authority/candidate/binding tuple. A family name or
/// target alone cannot select a fused module.
pub fn ApprovedFusedKernelCatalog::resolve_exact(
  self : ApprovedFusedKernelCatalog,
  manifest_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  target : @catalog.DeviceTarget,
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  toolchain_digest_sha256 : String,
  driver_identity_sha256 : String,
  symbol : String,
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
  launch : @launch_contract.AotLaunchDimensions,
  module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  compiled_binding : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> ApprovedFusedKernelModule raise FusedArtifactAdmissionError {
  let admitted_manifest = self.manifest
  fused_require_catalog_authority(
    admitted_manifest,
    self.target,
    manifest_digest,
    target,
  )
  let entry = match family {
    QkvPositionedRopePagedKvWrite => self.qkv
    ResidualRmsNorm => self.residual
  }
  fused_require_exact_selection(
    entry, family, candidate, source, recipe, toolchain_digest_sha256, driver_identity_sha256,
    symbol, abi, launch, module_digest, compiled_binding,
  )
  entry
}
