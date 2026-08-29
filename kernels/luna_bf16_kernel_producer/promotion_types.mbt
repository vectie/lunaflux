///|
pub(all) enum LunaBf16PromotionRule {
  Limits
  Digest
  Shape
  Observation
  ObservationOrder
  CandidateCoverage
  TrialCoverage
  Correctness
  MicrobenchmarkWin
  EndToEndRegression
  CandidateSet
  ProfileCapture
  Target
  Compiler
  LaunchContract
  ReleaseAuthority
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror LunaBf16PromotionError {
  LunaBf16PromotionRejected(LunaBf16PromotionRule)
} derive(Debug, Eq)

///|
pub struct LunaBf16PromotionDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaBf16PromotionDigest::from_sha256(
  value : String,
) -> LunaBf16PromotionDigest raise LunaBf16PromotionError {
  if value.length() != 64 {
    raise LunaBf16PromotionRejected(Digest)
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      raise LunaBf16PromotionRejected(Digest)
    }
  }
  LunaBf16PromotionDigest(value)
}

///|
pub fn LunaBf16PromotionDigest::as_hex(
  self : LunaBf16PromotionDigest,
) -> String {
  self.0
}

///|
pub struct LunaBf16ProfilerEvidenceLimits {
  priv maximum_candidates : Int
  priv maximum_shapes_per_candidate : Int
  priv trials_per_shape : Int
  priv end_to_end_trials : Int
  priv maximum_end_to_end_regression_ppm : Int
  priv maximum_canonical_bytes : Int
}

///|
pub fn LunaBf16ProfilerEvidenceLimits::new(
  maximum_candidates~ : Int,
  maximum_shapes_per_candidate~ : Int,
  trials_per_shape~ : Int,
  end_to_end_trials~ : Int,
  maximum_end_to_end_regression_ppm~ : Int,
  maximum_canonical_bytes~ : Int,
) -> LunaBf16ProfilerEvidenceLimits raise LunaBf16PromotionError {
  if maximum_candidates <= 0 ||
    maximum_candidates > 4096 ||
    maximum_shapes_per_candidate <= 0 ||
    maximum_shapes_per_candidate > 1024 ||
    trials_per_shape <= 0 ||
    trials_per_shape > 1024 ||
    end_to_end_trials <= 0 ||
    end_to_end_trials > 1024 ||
    maximum_end_to_end_regression_ppm < 0 ||
    maximum_end_to_end_regression_ppm > 1000000 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16777216 {
    raise LunaBf16PromotionRejected(Limits)
  }
  {
    maximum_candidates,
    maximum_shapes_per_candidate,
    trials_per_shape,
    end_to_end_trials,
    maximum_end_to_end_regression_ppm,
    maximum_canonical_bytes,
  }
}

///|
pub struct LunaBf16BenchmarkShape {
  priv prefill_rows : Int
  priv decode_rows : Int
  priv query_tokens : Int
  priv page_table_entries : Int
  priv trace : LunaBf16PromotionDigest
} derive(Debug, Eq)

///|
pub fn LunaBf16BenchmarkShape::new(
  prefill_rows~ : Int,
  decode_rows~ : Int,
  query_tokens~ : Int,
  page_table_entries~ : Int,
  trace~ : LunaBf16PromotionDigest,
) -> LunaBf16BenchmarkShape raise LunaBf16PromotionError {
  if prefill_rows < 0 ||
    decode_rows < 0 ||
    prefill_rows + decode_rows <= 0 ||
    query_tokens <= 0 ||
    page_table_entries <= 0 {
    raise LunaBf16PromotionRejected(Shape)
  }
  { prefill_rows, decode_rows, query_tokens, page_table_entries, trace }
}

///|
pub struct LunaBf16MicrobenchmarkObservation {
  priv operation : @plan.OperationId
  priv shape_ordinal : Int
  priv trial_ordinal : Int
  priv shape : LunaBf16BenchmarkShape
  priv baseline_duration_ns : Int64
  priv candidate_duration_ns : Int64
  priv correctness : LunaBf16PromotionDigest
  priv dispatch_canary : LunaBf16PromotionDigest
  priv correctness_passed : Bool
}

///|
pub fn LunaBf16MicrobenchmarkObservation::new(
  operation~ : @plan.OperationId,
  shape_ordinal~ : Int,
  trial_ordinal~ : Int,
  shape~ : LunaBf16BenchmarkShape,
  baseline_duration_ns~ : Int64,
  candidate_duration_ns~ : Int64,
  correctness~ : LunaBf16PromotionDigest,
  dispatch_canary~ : LunaBf16PromotionDigest,
  correctness_passed~ : Bool,
) -> LunaBf16MicrobenchmarkObservation raise LunaBf16PromotionError {
  if shape_ordinal < 0 ||
    trial_ordinal < 0 ||
    baseline_duration_ns <= 0L ||
    baseline_duration_ns > 1000000000000L ||
    candidate_duration_ns <= 0L ||
    candidate_duration_ns > 1000000000000L {
    raise LunaBf16PromotionRejected(Observation)
  }
  {
    operation,
    shape_ordinal,
    trial_ordinal,
    shape,
    baseline_duration_ns,
    candidate_duration_ns,
    correctness,
    dispatch_canary,
    correctness_passed,
  }
}

///|
pub struct LunaBf16EndToEndObservation {
  priv trial_ordinal : Int
  priv workload : LunaBf16PromotionDigest
  priv baseline_duration_ns : Int64
  priv candidate_duration_ns : Int64
  priv correctness : LunaBf16PromotionDigest
  priv correctness_passed : Bool
}

///|
pub fn LunaBf16EndToEndObservation::new(
  trial_ordinal~ : Int,
  workload~ : LunaBf16PromotionDigest,
  baseline_duration_ns~ : Int64,
  candidate_duration_ns~ : Int64,
  correctness~ : LunaBf16PromotionDigest,
  correctness_passed~ : Bool,
) -> LunaBf16EndToEndObservation raise LunaBf16PromotionError {
  if trial_ordinal < 0 ||
    baseline_duration_ns <= 0L ||
    baseline_duration_ns > 1000000000000L ||
    candidate_duration_ns <= 0L ||
    candidate_duration_ns > 1000000000000L {
    raise LunaBf16PromotionRejected(Observation)
  }
  {
    trial_ordinal,
    workload,
    baseline_duration_ns,
    candidate_duration_ns,
    correctness,
    correctness_passed,
  }
}

///|
/// Structurally admitted offline measurements. Timing and correctness facts
/// remain profiler/referee claims until independently reproduced and approved.
pub struct LunaBf16ProfilerEvidence {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv candidate_set_digest : String
  priv compiler_digest : LunaBf16PromotionDigest
  priv profile_capture_digest : @luna_profile_priority.LunaKernelProfileDigest
  priv microbenchmark_count : Int
  priv end_to_end_count : Int
  priv canonical : Bytes
  priv digest : LunaBf16PromotionDigest
}

///|
pub fn LunaBf16ProfilerEvidence::digest(
  self : LunaBf16ProfilerEvidence,
) -> LunaBf16PromotionDigest {
  self.digest
}

///|
pub fn LunaBf16ProfilerEvidence::canonical_bytes(
  self : LunaBf16ProfilerEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaBf16ProfilerEvidence::microbenchmark_count(
  self : LunaBf16ProfilerEvidence,
) -> Int {
  self.microbenchmark_count
}

///|
pub fn LunaBf16ProfilerEvidence::end_to_end_count(
  self : LunaBf16ProfilerEvidence,
) -> Int {
  self.end_to_end_count
}

///|
pub struct LunaBf16PromotionLimits {
  priv maximum_canonical_bytes : Int
}

///|
pub fn LunaBf16PromotionLimits::new(
  maximum_canonical_bytes~ : Int,
) -> LunaBf16PromotionLimits raise LunaBf16PromotionError {
  if maximum_canonical_bytes <= 0 || maximum_canonical_bytes > 16777216 {
    raise LunaBf16PromotionRejected(Limits)
  }
  { maximum_canonical_bytes, }
}

///|
/// Opaque software promotion join. It is not deployment approval, physical
/// benchmark reproduction, device authority, execution authority, or readiness.
pub struct LunaBf16OfflinePromotion {
  priv digest : LunaBf16PromotionDigest
  priv profiler_evidence_digest : LunaBf16PromotionDigest
  priv candidate_set_digest : String
  priv release_manifest_digest : String
  priv launch_contract_digest : LunaBf16PromotionDigest
  priv operation_count : Int
}

///|
pub fn LunaBf16OfflinePromotion::digest(
  self : LunaBf16OfflinePromotion,
) -> LunaBf16PromotionDigest {
  self.digest
}

///|
pub fn LunaBf16OfflinePromotion::profiler_evidence_digest(
  self : LunaBf16OfflinePromotion,
) -> LunaBf16PromotionDigest {
  self.profiler_evidence_digest
}

///|
pub fn LunaBf16OfflinePromotion::operation_count(
  self : LunaBf16OfflinePromotion,
) -> Int {
  self.operation_count
}

///|
pub fn LunaBf16OfflinePromotion::candidate_set_digest(
  self : LunaBf16OfflinePromotion,
) -> String {
  self.candidate_set_digest
}

///|
pub fn LunaBf16OfflinePromotion::release_manifest_digest(
  self : LunaBf16OfflinePromotion,
) -> String {
  self.release_manifest_digest
}

///|
pub fn LunaBf16OfflinePromotion::launch_contract_digest(
  self : LunaBf16OfflinePromotion,
) -> LunaBf16PromotionDigest {
  self.launch_contract_digest
}
