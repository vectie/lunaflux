///|
/// Exact immutable record proposed to the deployment trust boundary. It binds
/// both compiled artifacts to physical correctness and paired benchmark
/// evidence, but remains inert until its canonical bytes are externally
/// approved.
pub struct FusedManifestApprovalRecord {
  priv physical : AdmittedFusedPhysicalEvidence
  priv qkv_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  priv residual_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  priv qkv_benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence
  priv residual_benchmark : @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence
  priv canonical : Bytes
  priv digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
}

///|
/// Manifest authority created only after the deployment-owned detached
/// signature receipt matches the exact approval record. The underlying
/// candidate and evidence types retain their original non-bindable semantics.
/// No production manifest loader, kernel bundle, worker, service, or runtime
/// route accepts this wrapper yet.
pub struct ApprovedFusedManifestArtifacts {
  priv record : FusedManifestApprovalRecord
  priv approval : @luna_capability_manifest.LunaAdmittedExternalRecord
}
