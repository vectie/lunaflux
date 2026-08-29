///|
fn fused_require_exact_selection(
  entry : ApprovedFusedKernelModule,
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
) -> Unit raise FusedArtifactAdmissionError {
  if entry.family != family {
    raise MismatchedFusedArtifact(Family)
  }
  if entry.candidate != candidate ||
    entry.source != source ||
    entry.recipe != recipe {
    raise MismatchedFusedArtifact(Candidate)
  }
  if entry.toolchain != toolchain_digest_sha256 ||
    entry.driver != driver_identity_sha256 {
    raise MismatchedFusedArtifact(Compiler)
  }
  if entry.symbol != symbol {
    raise MismatchedFusedArtifact(Symbol)
  }
  if entry.abi != abi {
    raise MismatchedFusedArtifact(KernelAbi)
  }
  match (entry.purpose, abi) {
    (QualificationDiagnostics, ResidualRmsNormProductionAbiV2)
    | (QualificationDiagnostics, QkvPositionedRopePagedKvWriteProductionAbiV2)
    | (ProductionFastPath, ResidualRmsNormAbiV1)
    | (ProductionFastPath, QkvPositionedRopePagedKvWriteAbiV1) =>
      raise MismatchedFusedArtifact(KernelAbi)
    _ => ()
  }
  if entry.launch != launch {
    raise MismatchedFusedArtifact(Launch)
  }
  if entry.module_digest != module_digest {
    raise MismatchedFusedArtifact(Module)
  }
  if entry.binding_digest != compiled_binding {
    raise MismatchedFusedArtifact(Binding)
  }
}

///|
fn fused_require_catalog_authority(
  admitted_manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  admitted_target : @catalog.DeviceTarget,
  requested_manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  requested_target : @catalog.DeviceTarget,
) -> Unit raise FusedArtifactAdmissionError {
  if requested_manifest != admitted_manifest {
    raise MismatchedFusedArtifact(Manifest)
  }
  if requested_target != admitted_target {
    raise MismatchedFusedArtifact(Target)
  }
}
