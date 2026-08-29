///|
pub struct FusedParallelPromotionTrial {
  priv shape_ordinal : Int
  priv trial_ordinal : Int
  priv baseline_duration_ns : Int64
  priv candidate_duration_ns : Int64
  priv maximum_absolute_error_ppb : Int64
  priv maximum_relative_error_ppb : Int64
  priv correctness : FusedParallelDigest
  priv dispatch_canary : FusedParallelDigest
}

///|
pub fn FusedParallelPromotionTrial::new(
  shape_ordinal~ : Int,
  trial_ordinal~ : Int,
  baseline_duration_ns~ : Int64,
  candidate_duration_ns~ : Int64,
  maximum_absolute_error_ppb~ : Int64,
  maximum_relative_error_ppb~ : Int64,
  correctness~ : FusedParallelDigest,
  dispatch_canary~ : FusedParallelDigest,
) -> FusedParallelPromotionTrial raise FusedParallelAotError {
  if shape_ordinal < 0 ||
    trial_ordinal < 0 ||
    baseline_duration_ns <= 0L ||
    baseline_duration_ns > 1000000000000L ||
    candidate_duration_ns <= 0L ||
    candidate_duration_ns > 1000000000000L ||
    maximum_absolute_error_ppb < 0L ||
    maximum_relative_error_ppb < 0L ||
    correctness == dispatch_canary {
    raise FusedParallelInvalid(Evidence)
  }
  {
    shape_ordinal,
    trial_ordinal,
    baseline_duration_ns,
    candidate_duration_ns,
    maximum_absolute_error_ppb,
    maximum_relative_error_ppb,
    correctness,
    dispatch_canary,
  }
}

///|
pub struct FusedParallelPromotionLimits {
  priv maximum_shapes : Int
  priv trials_per_shape : Int
  priv maximum_canonical_bytes : Int
}

///|
pub fn FusedParallelPromotionLimits::new(
  maximum_shapes~ : Int,
  trials_per_shape~ : Int,
  maximum_canonical_bytes~ : Int,
) -> FusedParallelPromotionLimits raise FusedParallelAotError {
  if maximum_shapes <= 0 ||
    maximum_shapes > 1024 ||
    trials_per_shape <= 0 ||
    trials_per_shape > 1024 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16777216 {
    raise FusedParallelInvalid(Limits)
  }
  { maximum_shapes, trials_per_shape, maximum_canonical_bytes }
}

///|
/// Structurally admitted caller-supplied offline observations. This value is
/// candidate-only: it is not physical reproduction, release approval, or a
/// manifest binding.
pub struct FusedParallelPromotionEvidence {
  priv candidate_digest : FusedParallelDigest
  priv static_resource_audit : FusedParallelDigest
  priv trial_count : Int
  priv canonical : Bytes
  priv digest : FusedParallelDigest
}

///|
pub fn FusedParallelPromotionEvidence::candidate_digest(
  self : FusedParallelPromotionEvidence,
) -> FusedParallelDigest {
  self.candidate_digest
}

///|
pub fn FusedParallelPromotionEvidence::static_resource_audit(
  self : FusedParallelPromotionEvidence,
) -> FusedParallelDigest {
  self.static_resource_audit
}

///|
pub fn FusedParallelPromotionEvidence::trial_count(
  self : FusedParallelPromotionEvidence,
) -> Int {
  self.trial_count
}

///|
pub fn FusedParallelPromotionEvidence::canonical_bytes(
  self : FusedParallelPromotionEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn FusedParallelPromotionEvidence::digest(
  self : FusedParallelPromotionEvidence,
) -> FusedParallelDigest {
  self.digest
}

///|
pub fn FusedParallelPromotionEvidence::manifest_bindable(
  _self : FusedParallelPromotionEvidence,
) -> Bool {
  false
}

///|
fn admit_fused_parallel_promotion_evidence(
  family : FusedParallelFamily,
  candidate_digest : FusedParallelDigest,
  source_digest : FusedParallelDigest,
  recipe_digest : FusedParallelDigest,
  policy : FusedParallelNumericalPolicy,
  static_resource_audit : FusedParallelDigest,
  trials : ArrayView[FusedParallelPromotionTrial],
  limits : FusedParallelPromotionLimits,
) -> FusedParallelPromotionEvidence raise FusedParallelAotError {
  if static_resource_audit == candidate_digest ||
    static_resource_audit == source_digest ||
    static_resource_audit == recipe_digest ||
    trials.is_empty() ||
    trials.length() % limits.trials_per_shape != 0 ||
    trials.length() / limits.trials_per_shape > limits.maximum_shapes {
    raise FusedParallelMismatch(Evidence)
  }
  let mut shape = -1
  let mut trial = -1
  let mut baseline_total = 0L
  let mut candidate_total = 0L
  for observation in trials {
    if observation.shape_ordinal != shape {
      if shape >= 0 &&
        (
          trial != limits.trials_per_shape - 1 ||
          candidate_total >= baseline_total
        ) {
        raise FusedParallelMismatch(MicrobenchmarkWin)
      }
      if observation.shape_ordinal != shape + 1 ||
        observation.trial_ordinal != 0 {
        raise FusedParallelMismatch(ObservationOrder)
      }
      shape = observation.shape_ordinal
      trial = -1
      baseline_total = 0L
      candidate_total = 0L
    }
    if observation.trial_ordinal != trial + 1 ||
      observation.trial_ordinal >= limits.trials_per_shape {
      raise FusedParallelMismatch(ObservationOrder)
    }
    if observation.maximum_absolute_error_ppb >
      policy.maximum_absolute_error_ppb() ||
      observation.maximum_relative_error_ppb >
      policy.maximum_relative_error_ppb() {
      raise FusedParallelMismatch(Correctness)
    }
    if baseline_total > 9223372036854775807L - observation.baseline_duration_ns ||
      candidate_total > 9223372036854775807L - observation.candidate_duration_ns {
      raise FusedParallelOverflow(Evidence)
    }
    baseline_total += observation.baseline_duration_ns
    candidate_total += observation.candidate_duration_ns
    trial = observation.trial_ordinal
  }
  if trial != limits.trials_per_shape - 1 || candidate_total >= baseline_total {
    raise FusedParallelMismatch(MicrobenchmarkWin)
  }
  let family_name = match family {
    QkvPositionedRopePagedKvWrite => "qkv-positioned-rope-paged-kvwrite"
    ResidualRmsNorm => "residual-rmsnorm"
  }
  let output = StringBuilder()
  output.write_string("schema=lunaflux-fused-parallel-raw-evidence.v1\n")
  output.write_string("family=" + family_name + "\n")
  output.write_string("candidate_sha256=" + candidate_digest.as_hex() + "\n")
  output.write_string("source_sha256=" + source_digest.as_hex() + "\n")
  output.write_string("recipe_sha256=" + recipe_digest.as_hex() + "\n")
  output.write_string("numerics=" + policy.name() + "\n")
  output.write_string(
    "static_resource_audit_sha256=" + static_resource_audit.as_hex() + "\n",
  )
  output.write_string("trial_count=" + trials.length().to_string() + "\n")
  for observation in trials {
    output.write_string(
      observation.shape_ordinal.to_string() +
      "," +
      observation.trial_ordinal.to_string() +
      "," +
      observation.baseline_duration_ns.to_string() +
      "," +
      observation.candidate_duration_ns.to_string() +
      "," +
      observation.maximum_absolute_error_ppb.to_string() +
      "," +
      observation.maximum_relative_error_ppb.to_string() +
      "," +
      observation.correctness.as_hex() +
      "," +
      observation.dispatch_canary.as_hex() +
      "\n",
    )
  }
  output.write_string("measurement_authority=caller-supplied-raw-claim\n")
  output.write_string("physical_performance_authority=external\n")
  output.write_string("manifest_bindable=false\n")
  let canonical = @utf8.encode(output.to_string())
  if canonical.length() > limits.maximum_canonical_bytes {
    raise FusedParallelLimitExceeded(Evidence)
  }
  {
    candidate_digest,
    static_resource_audit,
    trial_count: trials.length(),
    canonical,
    digest: fused_parallel_digest(canonical),
  }
}

///|
pub fn admit_fused_qkv_rope_kv_promotion_evidence(
  candidate : FusedQkvRopeKvWriteCudaCandidate,
  static_resource_audit : FusedParallelDigest,
  trials : ArrayView[FusedParallelPromotionTrial],
  limits : FusedParallelPromotionLimits,
) -> FusedParallelPromotionEvidence raise FusedParallelAotError {
  admit_fused_parallel_promotion_evidence(
    QkvPositionedRopePagedKvWrite,
    candidate.candidate_digest(),
    candidate.source_digest(),
    candidate.recipe_digest(),
    candidate.numerical_policy(),
    static_resource_audit,
    trials,
    limits,
  )
}

///|
pub fn admit_fused_residual_rmsnorm_promotion_evidence(
  candidate : FusedResidualRmsNormCudaCandidate,
  static_resource_audit : FusedParallelDigest,
  trials : ArrayView[FusedParallelPromotionTrial],
  limits : FusedParallelPromotionLimits,
) -> FusedParallelPromotionEvidence raise FusedParallelAotError {
  admit_fused_parallel_promotion_evidence(
    ResidualRmsNorm,
    candidate.candidate_digest(),
    candidate.source_digest(),
    candidate.recipe_digest(),
    candidate.numerical_policy(),
    static_resource_audit,
    trials,
    limits,
  )
}
