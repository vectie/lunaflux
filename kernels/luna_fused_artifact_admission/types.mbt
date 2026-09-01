///|
pub(all) enum FusedArtifactAdmissionRule {
  Manifest
  Family
  Candidate
  Compiler
  Target
  Launch
  Symbol
  KernelAbi
  Module
  Receipt
  Binding
  Fallback
  Catalog
} derive(Debug, Eq)

///|
pub(all) suberror FusedArtifactAdmissionError {
  InvalidFusedArtifact(FusedArtifactAdmissionRule)
  MismatchedFusedArtifact(FusedArtifactAdmissionRule)
  MissingFusedArtifact(FusedArtifactAdmissionRule)
} derive(Debug, Eq)

///|
pub(all) enum FusedStandaloneFallbackRole {
  QkvProjectionFallback
  PositionedRopeFallback
  PagedKvWriteFallback
  ResidualAddFallback
  RmsNormFallback
} derive(Debug, Eq)

///|
/// Content identity of one standalone fallback. It grants no module or launch
/// authority and is safe to construct before external release approval.
pub struct FusedStandaloneFallbackIdentity {
  priv role : FusedStandaloneFallbackRole
  priv operation : Int
  priv source : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest
} derive(Debug, Eq)

///|
pub fn FusedStandaloneFallbackIdentity::new(
  role~ : FusedStandaloneFallbackRole,
  operation~ : Int,
  source~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> FusedStandaloneFallbackIdentity raise FusedArtifactAdmissionError {
  if operation < 0 {
    raise InvalidFusedArtifact(Fallback)
  }
  { role, operation, source, recipe }
}

///|
/// Exact, content-addressed module retained by the approved fused bundle.
/// This type has no public constructor and is not independently executable.
pub struct ApprovedFusedKernelModule {
  priv family : @luna_cuda_fused_parallel_aot.FusedParallelFamily
  priv candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv source : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv toolchain : String
  priv driver : String
  priv target : @catalog.DeviceTarget
  priv launch : @launch_contract.AotLaunchDimensions
  priv symbol : String
  priv abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi
  priv purpose : @luna_cuda_fused_parallel_aot.FusedParallelArtifactPurpose
  priv dispatch_canary_per_token : UInt
  priv artifact_bytes : Bytes
  priv module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv receipt_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv binding_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv benchmark_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity]
}

///|
pub struct ApprovedFusedKernelBundle {
  priv manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv target : @catalog.DeviceTarget
  priv qkv : ApprovedFusedKernelModule
  priv residual : ApprovedFusedKernelModule
}

///|
/// Exact catalog derived directly from the approved production manifest.
/// The residual entry carries an explicit qualification/production purpose;
/// exact selection includes its ABI so a diagnostic module cannot satisfy a
/// production request.
pub struct ApprovedFusedKernelCatalog {
  priv manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv target : @catalog.DeviceTarget
  priv qkv : ApprovedFusedKernelModule
  priv residual : ApprovedFusedKernelModule
  priv canonical : Bytes
  priv digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
}

///|
/// Digest-pinned, shape-reusable production residual/RMSNorm module. One seed
/// pair binds its exact model shape; device-step admission may reuse the same
/// AOT function for identical pairs in that model without token-step checks.
pub struct ApprovedReusableFusedResidualRuntime {
  priv source_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv model_content : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv model_plan : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv target : @catalog.DeviceTarget
  priv residual_operation : Int
  priv norm_operation : Int
  priv symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv module_bytes : Bytes
}

///|
/// Production ABI vocabulary for the bounded reusable runtime bundle. Each
/// variant names one exact semantic family; incompatible or future ABIs must
/// be admitted by a new schema rather than reinterpreted silently.
pub(all) enum ReusableFusedRuntimeKernelAbi {
  ResidualRmsNormProductionV2
  QwenQkNormPositionedRopePagedKvWriteProductionV1
  QwenPagedAttentionReadOnlyProductionV2
} derive(Debug, Eq)

///|
/// Startup-side module input used only to export a digest-pinned runtime
/// bundle. It is not executable authority. The constructor enforces the exact
/// contiguous operation span required by the ABI before canonicalization.
pub struct ReusableFusedRuntimeModuleInput {
  priv abi : ReusableFusedRuntimeKernelAbi
  priv first_operation : Int
  priv operation_count : Int
  priv symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv module_bytes : Bytes
}

///|
/// One module admitted from the exact digest-pinned runtime bundle. Runtime
/// owners may extract these immutable fields during startup; token execution
/// never parses or revalidates the canonical sidecar.
pub struct ApprovedReusableFusedRuntimeModule {
  priv abi : ReusableFusedRuntimeKernelAbi
  priv first_operation : Int
  priv operation_count : Int
  priv symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv module_bytes : Bytes
}

///|
/// Bounded multi-family startup admission. At most one module for each known
/// ABI is retained, so lookup and later runtime materialization have a fixed
/// upper bound independent of requests or model depth.
pub struct ApprovedReusableFusedRuntimeBundle {
  priv source_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv model_content : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv model_plan : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv target : @catalog.DeviceTarget
  priv modules : ReadOnlyArray[ApprovedReusableFusedRuntimeModule]
}
