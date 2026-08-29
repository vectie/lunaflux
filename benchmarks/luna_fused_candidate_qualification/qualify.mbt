///|
fn fused_benchmark_qualify(
  identity : FusedBenchmarkCandidateIdentity,
  protocol : FusedCandidateBenchmarkProtocol,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
  maximum_evidence_bytes : Int,
) -> FusedCandidateBenchmarkQualificationEvidence raise FusedCandidateBenchmarkError {
  if maximum_evidence_bytes <= 0 || maximum_evidence_bytes > 16777216 {
    raise InvalidFusedCandidateBenchmark(EvidenceSize)
  }
  fused_benchmark_validate_protocol(identity, protocol)
  let summaries = fused_benchmark_validate_observations(
    identity, protocol, observations,
  )
  let canonical = fused_benchmark_canonical(
    identity, protocol, observations, summaries,
  )
  if canonical.length() > maximum_evidence_bytes {
    raise FusedCandidateBenchmarkLimitExceeded(EvidenceSize)
  }
  {
    family: identity.family,
    candidate: identity.candidate,
    shape_set: protocol.shape_set,
    observation_count: observations.length(),
    canonical,
    digest: @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ) catch {
      _ => abort("SHA-256 produced an invalid fused benchmark evidence digest")
    },
  }
}

///|
/// Structurally qualifies raw paired observations for the exact fused QKV,
/// positioned-RoPE, and paged-KV-write candidate. This produces inert evidence
/// only and does not assert that a runner or physical device produced the data.
pub fn qualify_fused_qkv_candidate_benchmark(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  protocol : FusedCandidateBenchmarkProtocol,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
  maximum_evidence_bytes~ : Int,
) -> FusedCandidateBenchmarkQualificationEvidence raise FusedCandidateBenchmarkError {
  fused_benchmark_qualify(
    fused_benchmark_qkv_identity(candidate),
    protocol,
    observations,
    maximum_evidence_bytes,
  )
}

///|
/// Structurally qualifies raw paired observations for the exact fused residual
/// plus RMSNorm candidate. No manifest, runtime, or promotion authority is
/// created.
pub fn qualify_fused_residual_rmsnorm_candidate_benchmark(
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  protocol : FusedCandidateBenchmarkProtocol,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
  maximum_evidence_bytes~ : Int,
) -> FusedCandidateBenchmarkQualificationEvidence raise FusedCandidateBenchmarkError {
  fused_benchmark_qualify(
    fused_benchmark_residual_identity(candidate),
    protocol,
    observations,
    maximum_evidence_bytes,
  )
}
