///|
fn validate_promotion_shape(
  shape : LunaBf16BenchmarkShape,
  profile : @launch_contract.PagedKernelProfile,
) -> Unit raise LunaBf16PromotionError {
  if shape.prefill_rows + shape.decode_rows > profile.max_query_rows() ||
    shape.query_tokens > profile.max_query_tokens() ||
    shape.page_table_entries > profile.max_page_table_entries() {
    raise LunaBf16PromotionRejected(Shape)
  }
}

///|
fn promotion_checked_add(
  left : Int64,
  right : Int64,
) -> Int64 raise LunaBf16PromotionError {
  if right <= 0L || left > 9223372036854775807L - right {
    raise LunaBf16PromotionRejected(Observation)
  }
  left + right
}

///|
fn validate_microbenchmark_evidence(
  candidates : LunaBf16CandidateSet,
  observations : ArrayView[LunaBf16MicrobenchmarkObservation],
  limits : LunaBf16ProfilerEvidenceLimits,
) -> Unit raise LunaBf16PromotionError {
  let candidate_count = candidates.operation_count()
  if candidate_count <= 0 ||
    candidate_count > limits.maximum_candidates ||
    observations.is_empty() ||
    observations.length() % limits.trials_per_shape != 0 ||
    observations.length() / limits.trials_per_shape >
    limits.maximum_candidates * limits.maximum_shapes_per_candidate ||
    observations.length() > limits.maximum_canonical_bytes / 64 {
    raise LunaBf16PromotionRejected(CandidateCoverage)
  }
  let mut candidate_index = -1
  let mut shape_index = -1
  let mut trial_index = -1
  let mut active_shape : LunaBf16BenchmarkShape? = None
  let mut baseline_total = 0L
  let mut candidate_total = 0L
  let close_shape = () => {
    if trial_index != limits.trials_per_shape - 1 ||
      candidate_total >= baseline_total {
      raise LunaBf16PromotionRejected(
        if trial_index != limits.trials_per_shape - 1 {
          TrialCoverage
        } else {
          MicrobenchmarkWin
        },
      )
    }
  }
  for observation in observations {
    let operation_index = observation.operation.as_int()
    if operation_index < 0 ||
      operation_index >= candidate_count ||
      candidates.candidates()[operation_index].operation_id() !=
      observation.operation {
      raise LunaBf16PromotionRejected(CandidateSet)
    }
    let changed = operation_index != candidate_index ||
      observation.shape_ordinal != shape_index
    if changed {
      if candidate_index >= 0 {
        close_shape()
      }
      if operation_index == candidate_index {
        if observation.shape_ordinal != shape_index + 1 {
          raise LunaBf16PromotionRejected(ObservationOrder)
        }
      } else if operation_index != candidate_index + 1 ||
        observation.shape_ordinal != 0 {
        raise LunaBf16PromotionRejected(ObservationOrder)
      }
      candidate_index = operation_index
      shape_index = observation.shape_ordinal
      if shape_index >= limits.maximum_shapes_per_candidate {
        raise LunaBf16PromotionRejected(CandidateCoverage)
      }
      trial_index = -1
      active_shape = Some(observation.shape)
      baseline_total = 0L
      candidate_total = 0L
    }
    guard active_shape is Some(expected_shape) else {
      raise LunaBf16PromotionRejected(ObservationOrder)
    }
    if observation.shape != expected_shape ||
      observation.trial_ordinal != trial_index + 1 ||
      observation.trial_ordinal >= limits.trials_per_shape {
      raise LunaBf16PromotionRejected(ObservationOrder)
    }
    validate_promotion_shape(observation.shape, candidates.profile())
    if !observation.correctness_passed {
      raise LunaBf16PromotionRejected(Correctness)
    }
    baseline_total = promotion_checked_add(
      baseline_total,
      observation.baseline_duration_ns,
    )
    candidate_total = promotion_checked_add(
      candidate_total,
      observation.candidate_duration_ns,
    )
    trial_index = observation.trial_ordinal
  }
  close_shape()
  if candidate_index != candidate_count - 1 {
    raise LunaBf16PromotionRejected(CandidateCoverage)
  }
}

///|
fn validate_end_to_end_evidence(
  profile : @luna_profile_priority.LunaPagedKernelProfileCapture,
  observations : ArrayView[LunaBf16EndToEndObservation],
  limits : LunaBf16ProfilerEvidenceLimits,
) -> Unit raise LunaBf16PromotionError {
  if observations.length() != limits.end_to_end_trials {
    raise LunaBf16PromotionRejected(TrialCoverage)
  }
  for index, observation in observations {
    if observation.trial_ordinal != index ||
      observation.workload.as_hex() != profile.workload_digest().as_hex() {
      raise LunaBf16PromotionRejected(ObservationOrder)
    }
    if !observation.correctness_passed {
      raise LunaBf16PromotionRejected(Correctness)
    }
    let allowed = observation.baseline_duration_ns +
      observation.baseline_duration_ns *
      limits.maximum_end_to_end_regression_ppm.to_int64() /
      1000000L
    if observation.candidate_duration_ns > allowed {
      raise LunaBf16PromotionRejected(EndToEndRegression)
    }
  }
}

///|
/// Collects bounded caller-supplied offline profiler/referee observations into
/// deterministic evidence. It opens no device, profiler, compiler, or process.
pub fn collect_luna_bf16_profiler_evidence(
  candidates : LunaBf16CandidateSet,
  profile : @luna_profile_priority.LunaPagedKernelProfileCapture,
  microbenchmarks : ArrayView[LunaBf16MicrobenchmarkObservation],
  end_to_end : ArrayView[LunaBf16EndToEndObservation],
  limits : LunaBf16ProfilerEvidenceLimits,
) -> LunaBf16ProfilerEvidence raise LunaBf16PromotionError {
  if profile.identity() != candidates.model_identity() ||
    profile.target() != candidates.target() ||
    profile.contract().kernel_profile() != candidates.profile() {
    raise LunaBf16PromotionRejected(ProfileCapture)
  }
  let priority = profile.priority().operation_id().as_int()
  if priority < 0 ||
    priority >= candidates.operation_count() ||
    candidates.candidates()[priority].operation_id() !=
    profile.priority().operation_id() {
    raise LunaBf16PromotionRejected(ProfileCapture)
  }
  validate_microbenchmark_evidence(candidates, microbenchmarks, limits)
  validate_end_to_end_evidence(profile, end_to_end, limits)
  let canonical = encode_profiler_evidence(
    candidates, profile, microbenchmarks, end_to_end, limits,
  )
  {
    identity: candidates.model_identity(),
    target: candidates.target(),
    profile: candidates.profile(),
    candidate_set_digest: candidates.candidate_set_sha256(),
    compiler_digest: promotion_compiler_digest(candidates.compiler_policy()),
    profile_capture_digest: profile.digest(),
    microbenchmark_count: microbenchmarks.length(),
    end_to_end_count: end_to_end.length(),
    canonical,
    digest: LunaBf16PromotionDigest(sha256_bytes(canonical)),
  }
}
