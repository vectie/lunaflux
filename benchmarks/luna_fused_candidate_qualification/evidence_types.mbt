///|
priv struct FusedBenchmarkIntegerSummary {
  count : Int
  minimum : Int64
  p50 : Int64
  maximum : Int64
  total : Int64
}

///|
pub struct FusedCandidateBenchmarkQualificationEvidence {
  priv family : @luna_cuda_fused_parallel_aot.FusedParallelFamily
  priv candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv shape_set : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv observation_count : Int
  priv canonical : Bytes
  priv digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::family(
  self : FusedCandidateBenchmarkQualificationEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelFamily {
  self.family
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::candidate_digest(
  self : FusedCandidateBenchmarkQualificationEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.candidate
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::shape_set_identity(
  self : FusedCandidateBenchmarkQualificationEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.shape_set
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::observation_count(
  self : FusedCandidateBenchmarkQualificationEvidence,
) -> Int {
  self.observation_count
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::canonical_bytes(
  self : FusedCandidateBenchmarkQualificationEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::digest(
  self : FusedCandidateBenchmarkQualificationEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.digest
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::manifest_bindable(
  _self : FusedCandidateBenchmarkQualificationEvidence,
) -> Bool {
  false
}

///|
pub fn FusedCandidateBenchmarkQualificationEvidence::promotion_authority(
  _self : FusedCandidateBenchmarkQualificationEvidence,
) -> Bool {
  false
}
