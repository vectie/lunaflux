///|
fn fused_artifact_digest(
  bytes : Bytes,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise FusedArtifactAdmissionError {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  ) catch {
    _ => raise InvalidFusedArtifact(Module)
  }
}

///|
fn fused_expected_abi(
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  purpose : @luna_cuda_fused_parallel_aot.FusedParallelArtifactPurpose,
) -> @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi {
  match (family, purpose) {
    (QkvPositionedRopePagedKvWrite, QualificationDiagnostics) =>
      QkvPositionedRopePagedKvWriteAbiV1
    (ResidualRmsNorm, QualificationDiagnostics) => ResidualRmsNormAbiV1
    (ResidualRmsNorm, ProductionFastPath) => ResidualRmsNormProductionAbiV2
    (QkvPositionedRopePagedKvWrite, ProductionFastPath) =>
      QkvPositionedRopePagedKvWriteProductionAbiV2
  }
}

///|
fn fused_validate_fallbacks(
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity],
) -> Unit raise FusedArtifactAdmissionError {
  let roles = match family {
    QkvPositionedRopePagedKvWrite =>
      [QkvProjectionFallback, PositionedRopeFallback, PagedKvWriteFallback]
    ResidualRmsNorm => [ResidualAddFallback, RmsNormFallback]
  }
  if fallbacks.length() != roles.length() {
    raise MismatchedFusedArtifact(Fallback)
  }
  for index, role in roles {
    let fallback = fallbacks[index]
    if fallback.role() != role || fallback.operation_index() < 0 {
      raise MismatchedFusedArtifact(Fallback)
    }
  }
}

///|
fn fused_admit_module(
  compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  benchmark_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  dispatch_canary_per_token : UInt,
  fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity],
  target : @catalog.DeviceTarget,
) -> ApprovedFusedKernelModule raise FusedArtifactAdmissionError {
  if compiled.manifest_bindable() {
    raise MismatchedFusedArtifact(Manifest)
  }
  if compiled.target() != target {
    raise MismatchedFusedArtifact(Target)
  }
  if compiled.kernel_abi() !=
    fused_expected_abi(compiled.family(), compiled.artifact_purpose()) {
    raise MismatchedFusedArtifact(KernelAbi)
  }
  match compiled.artifact_purpose() {
    QualificationDiagnostics =>
      if dispatch_canary_per_token == 0U ||
        compiled.dispatch_canary_per_token() != dispatch_canary_per_token {
        raise MismatchedFusedArtifact(KernelAbi)
      }
    ProductionFastPath =>
      if dispatch_canary_per_token != 0U ||
        compiled.dispatch_canary_per_token() != 0U {
        raise MismatchedFusedArtifact(KernelAbi)
      }
  }
  if compiled.function_symbol().is_empty() {
    raise InvalidFusedArtifact(Symbol)
  }
  let artifact_bytes = compiled.module_bytes()
  if artifact_bytes.is_empty() ||
    fused_artifact_digest(artifact_bytes) != compiled.module_digest() {
    raise MismatchedFusedArtifact(Module)
  }
  if fused_artifact_digest(compiled.compile_receipt_bytes()) !=
    compiled.compile_receipt_digest() {
    raise MismatchedFusedArtifact(Receipt)
  }
  if fused_artifact_digest(compiled.binding_record_bytes()) !=
    compiled.binding_digest() {
    raise MismatchedFusedArtifact(Binding)
  }
  fused_validate_fallbacks(compiled.family(), fallbacks)
  {
    family: compiled.family(),
    candidate: compiled.candidate_digest(),
    source: compiled.source_digest(),
    recipe: compiled.recipe_digest(),
    toolchain: compiled.toolchain_digest_sha256(),
    driver: compiled.driver_identity_sha256(),
    target: compiled.target(),
    launch: compiled.launch_dimensions(),
    symbol: compiled.function_symbol(),
    abi: compiled.kernel_abi(),
    purpose: compiled.artifact_purpose(),
    dispatch_canary_per_token,
    artifact_bytes,
    module_digest: compiled.module_digest(),
    receipt_digest: compiled.compile_receipt_digest(),
    binding_digest: compiled.binding_digest(),
    benchmark_digest,
    fallbacks,
  }
}

///|
fn fused_admit_pair(
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
) -> (ApprovedFusedKernelModule, ApprovedFusedKernelModule) raise FusedArtifactAdmissionError {
  if approval.record_digest().as_hex() != approval_record_digest.as_hex() ||
    manifest_digest.as_hex().is_empty() {
    raise MismatchedFusedArtifact(Manifest)
  }
  let qkv = fused_admit_module(
    qkv_compiled, qkv_benchmark_digest, qkv_dispatch_canary_per_token, qkv_fallbacks,
    target,
  )
  let residual = fused_admit_module(
    residual_compiled, residual_benchmark_digest, residual_dispatch_canary_per_token,
    residual_fallbacks, target,
  )
  if qkv.family != QkvPositionedRopePagedKvWrite ||
    residual.family != ResidualRmsNorm {
    raise MismatchedFusedArtifact(Family)
  }
  if qkv.toolchain != residual.toolchain || qkv.driver != residual.driver {
    raise MismatchedFusedArtifact(Compiler)
  }
  (qkv, residual)
}

///|
/// Produces the immutable two-module bundle only from approved production
/// manifest authority. There is no candidate-only or evidence-only overload.
pub fn admit_approved_fused_kernel_bundle(
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
) -> ApprovedFusedKernelBundle raise FusedArtifactAdmissionError {
  let (qkv, residual) = fused_admit_pair(
    manifest_digest, approval_record_digest, target, qkv_compiled, residual_compiled,
    qkv_benchmark_digest, residual_benchmark_digest, qkv_dispatch_canary_per_token,
    residual_dispatch_canary_per_token, qkv_fallbacks, residual_fallbacks, approval,
  )
  { manifest: manifest_digest, target, qkv, residual }
}
