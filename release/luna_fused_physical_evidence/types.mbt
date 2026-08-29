///|
const FUSED_PHYSICAL_MAXIMUM_CANONICAL_BYTES : Int = 131072

///|
const FUSED_PHYSICAL_CANONICAL_LINE_COUNT : Int = 172

///|
const FUSED_PHYSICAL_REFEREE_SHAPE_COUNT : Int = 4

///|
const FUSED_PHYSICAL_REFEREE_TOKEN_COUNT : UInt = 12U

///|
priv struct FusedStandaloneFallbackIdentity {
  operation : Int
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest
}

///|
priv struct ExpectedFusedPhysicalFamily {
  family_name : String
  candidate_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  source_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  recipe_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  recipe_bytes : Bytes
  compiler : @luna_specialization.LunaCompilerPolicy
  target : @catalog.DeviceTarget
  profile : @launch_contract.PagedKernelProfile
  dimensions : @launch_contract.AotLaunchDimensions
  function_symbol : String
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi
  module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  receipt_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  driver_identity_sha256 : String
  numerical_policy : @luna_cuda_fused_parallel_aot.FusedParallelNumericalPolicy
  dispatch_canary : UInt
  fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity]
}

///|
priv struct AdmittedFusedPhysicalFamily {
  candidate_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  source_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  recipe_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  receipt_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  observed_absolute_error_ppb : Int64
  observed_relative_error_ppb : Int64
  dispatch_canary : UInt
  dispatch_canary_per_token : UInt
  fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity]
}

///|
/// Immutable, root-free observation that both exact fused Phase-5 candidates
/// satisfied the fixed physical qualification contract.
///
/// This value is intentionally inert. It grants no artifact loading, device,
/// process, compiler, runtime, manifest, deployment, or promotion authority.
pub struct AdmittedFusedPhysicalEvidence {
  priv canonical : Bytes
  priv digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv referee_shape_set_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv artifact_directory_seal_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv evidence_directory_seal_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv approved_policy_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv qkv : AdmittedFusedPhysicalFamily
  priv residual_norm : AdmittedFusedPhysicalFamily
}
