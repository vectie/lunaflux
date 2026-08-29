///|
suberror Fp8V2CompileCandidateError {
  InvalidArguments
  CandidateAdmission
  Publication
} derive(Debug, Eq)

///|
priv struct Fp8V2CompileCandidateSet {
  target : String
  model_content_sha256 : String
  model_plan_sha256 : String
  simple : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2
  gated : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2
  manifest : Bytes
  manifest_sha256 : String
}
