///|
pub fn ApprovedFusedKernelModule::family(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelFamily {
  self.family
}

///|
pub fn ApprovedFusedKernelModule::candidate_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.candidate
}

///|
pub fn ApprovedFusedKernelModule::source_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.source
}

///|
pub fn ApprovedFusedKernelModule::recipe_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.recipe
}

///|
pub fn ApprovedFusedKernelModule::toolchain_digest_sha256(
  self : ApprovedFusedKernelModule,
) -> String {
  self.toolchain
}

///|
pub fn ApprovedFusedKernelModule::driver_identity_sha256(
  self : ApprovedFusedKernelModule,
) -> String {
  self.driver
}

///|
pub fn ApprovedFusedKernelModule::target(
  self : ApprovedFusedKernelModule,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ApprovedFusedKernelModule::launch_dimensions(
  self : ApprovedFusedKernelModule,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn ApprovedFusedKernelModule::function_symbol(
  self : ApprovedFusedKernelModule,
) -> String {
  self.symbol
}

///|
pub fn ApprovedFusedKernelModule::kernel_abi(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi {
  self.abi
}

///|
pub fn ApprovedFusedKernelModule::artifact_purpose(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelArtifactPurpose {
  self.purpose
}

///|
pub fn ApprovedFusedKernelModule::dispatch_canary_per_token(
  self : ApprovedFusedKernelModule,
) -> UInt {
  self.dispatch_canary_per_token
}

///|
pub fn ApprovedFusedKernelModule::module_bytes(
  self : ApprovedFusedKernelModule,
) -> Bytes {
  self.artifact_bytes
}

///|
pub fn ApprovedFusedKernelModule::module_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.module_digest
}

///|
pub fn ApprovedFusedKernelModule::compile_receipt_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.receipt_digest
}

///|
pub fn ApprovedFusedKernelModule::compiled_binding_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.binding_digest
}

///|
pub fn ApprovedFusedKernelModule::benchmark_digest(
  self : ApprovedFusedKernelModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.benchmark_digest
}

///|
pub fn ApprovedFusedKernelModule::fallbacks(
  self : ApprovedFusedKernelModule,
) -> ReadOnlyArray[FusedStandaloneFallbackIdentity] {
  self.fallbacks
}

///|
pub fn FusedStandaloneFallbackIdentity::role(
  self : FusedStandaloneFallbackIdentity,
) -> FusedStandaloneFallbackRole {
  self.role
}

///|
pub fn FusedStandaloneFallbackIdentity::operation_index(
  self : FusedStandaloneFallbackIdentity,
) -> Int {
  self.operation
}

///|
pub fn FusedStandaloneFallbackIdentity::source_digest(
  self : FusedStandaloneFallbackIdentity,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.source
}

///|
pub fn FusedStandaloneFallbackIdentity::recipe_digest(
  self : FusedStandaloneFallbackIdentity,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.recipe
}

///|
pub fn ApprovedFusedKernelBundle::manifest_digest(
  self : ApprovedFusedKernelBundle,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.manifest
}

///|
pub fn ApprovedFusedKernelBundle::target(
  self : ApprovedFusedKernelBundle,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ApprovedFusedKernelBundle::qkv_module(
  self : ApprovedFusedKernelBundle,
) -> ApprovedFusedKernelModule {
  self.qkv
}

///|
pub fn ApprovedFusedKernelBundle::residual_module(
  self : ApprovedFusedKernelBundle,
) -> ApprovedFusedKernelModule {
  self.residual
}

///|
pub fn ApprovedFusedKernelCatalog::manifest_digest(
  self : ApprovedFusedKernelCatalog,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.manifest
}

///|
pub fn ApprovedFusedKernelCatalog::target(
  self : ApprovedFusedKernelCatalog,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ApprovedFusedKernelCatalog::canonical_bytes(
  self : ApprovedFusedKernelCatalog,
) -> Bytes {
  self.canonical
}

///|
pub fn ApprovedFusedKernelCatalog::digest(
  self : ApprovedFusedKernelCatalog,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.digest
}
