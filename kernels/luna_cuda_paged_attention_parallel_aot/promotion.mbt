///|
pub struct PagedAttentionParallelPromotionTrial {
  priv shape_ordinal : Int
  priv trial_ordinal : Int
  priv baseline_duration_ns : Int64
  priv candidate_duration_ns : Int64
  priv maximum_absolute_error_ppb : Int64
  priv maximum_relative_error_ppb : Int64
  priv token_agreement : Bool
  priv correctness : PagedAttentionParallelDigest
  priv dispatch_canary : PagedAttentionParallelDigest
}

///|
pub fn PagedAttentionParallelPromotionTrial::new(
  shape_ordinal~ : Int,
  trial_ordinal~ : Int,
  baseline_duration_ns~ : Int64,
  candidate_duration_ns~ : Int64,
  maximum_absolute_error_ppb~ : Int64,
  maximum_relative_error_ppb~ : Int64,
  token_agreement~ : Bool,
  correctness~ : PagedAttentionParallelDigest,
  dispatch_canary~ : PagedAttentionParallelDigest,
) -> PagedAttentionParallelPromotionTrial raise PagedAttentionParallelAotError {
  if shape_ordinal < 0 ||
    trial_ordinal < 0 ||
    baseline_duration_ns <= 0L ||
    baseline_duration_ns > 1000000000000L ||
    candidate_duration_ns <= 0L ||
    candidate_duration_ns > 1000000000000L ||
    maximum_absolute_error_ppb < 0L ||
    maximum_relative_error_ppb < 0L {
    raise PagedAttentionParallelInvalid(PromotionEvidence)
  }
  {
    shape_ordinal,
    trial_ordinal,
    baseline_duration_ns,
    candidate_duration_ns,
    maximum_absolute_error_ppb,
    maximum_relative_error_ppb,
    token_agreement,
    correctness,
    dispatch_canary,
  }
}

///|
pub struct PagedAttentionParallelEndToEndTrial {
  priv trial_ordinal : Int
  priv baseline_duration_ns : Int64
  priv candidate_duration_ns : Int64
  priv token_agreement : Bool
  priv correctness : PagedAttentionParallelDigest
}

///|
pub fn PagedAttentionParallelEndToEndTrial::new(
  trial_ordinal~ : Int,
  baseline_duration_ns~ : Int64,
  candidate_duration_ns~ : Int64,
  token_agreement~ : Bool,
  correctness~ : PagedAttentionParallelDigest,
) -> PagedAttentionParallelEndToEndTrial raise PagedAttentionParallelAotError {
  if trial_ordinal < 0 ||
    baseline_duration_ns <= 0L ||
    baseline_duration_ns > 1000000000000L ||
    candidate_duration_ns <= 0L ||
    candidate_duration_ns > 1000000000000L {
    raise PagedAttentionParallelInvalid(PromotionEvidence)
  }
  {
    trial_ordinal,
    baseline_duration_ns,
    candidate_duration_ns,
    token_agreement,
    correctness,
  }
}

///|
pub struct PagedAttentionParallelPromotionLimits {
  priv maximum_shapes : Int
  priv trials_per_shape : Int
  priv end_to_end_trials : Int
  priv maximum_end_to_end_regression_ppm : Int
  priv maximum_canonical_bytes : Int
}

///|
pub fn PagedAttentionParallelPromotionLimits::new(
  maximum_shapes~ : Int,
  trials_per_shape~ : Int,
  end_to_end_trials~ : Int,
  maximum_end_to_end_regression_ppm~ : Int,
  maximum_canonical_bytes~ : Int,
) -> PagedAttentionParallelPromotionLimits raise PagedAttentionParallelAotError {
  if maximum_shapes <= 0 ||
    maximum_shapes > 1024 ||
    trials_per_shape <= 0 ||
    trials_per_shape > 1024 ||
    end_to_end_trials <= 0 ||
    end_to_end_trials > 1024 ||
    maximum_end_to_end_regression_ppm < 0 ||
    maximum_end_to_end_regression_ppm > 1000000 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16777216 {
    raise PagedAttentionParallelInvalid(Limits)
  }
  {
    maximum_shapes,
    trials_per_shape,
    end_to_end_trials,
    maximum_end_to_end_regression_ppm,
    maximum_canonical_bytes,
  }
}

///|
/// Structurally admitted raw offline evidence. It is intentionally not a claim
/// that LunaFlux has reproduced a physical benchmark on any device.
pub struct PagedAttentionParallelPromotionEvidence {
  priv candidate_digest : PagedAttentionParallelDigest
  priv reference_source_digest : String
  priv trial_count : Int
  priv end_to_end_count : Int
  priv canonical : Bytes
  priv digest : PagedAttentionParallelDigest
}

///|
pub fn PagedAttentionParallelPromotionEvidence::digest(
  self : PagedAttentionParallelPromotionEvidence,
) -> PagedAttentionParallelDigest {
  self.digest
}

///|
pub fn PagedAttentionParallelPromotionEvidence::trial_count(
  self : PagedAttentionParallelPromotionEvidence,
) -> Int {
  self.trial_count
}

///|
pub fn PagedAttentionParallelPromotionEvidence::end_to_end_count(
  self : PagedAttentionParallelPromotionEvidence,
) -> Int {
  self.end_to_end_count
}

///|
pub fn PagedAttentionParallelPromotionEvidence::canonical_bytes(
  self : PagedAttentionParallelPromotionEvidence,
) -> Bytes {
  self.canonical
}

///|
fn promotion_append(
  output : StringBuilder,
  written : Ref[Int],
  value : String,
  maximum : Int,
) -> Unit raise PagedAttentionParallelAotError {
  let encoded = @utf8.encode(value)
  if encoded.length() >= maximum || written.val > maximum - encoded.length() - 1 {
    raise PagedAttentionParallelLimitExceeded(PromotionEvidence)
  }
  output.write_string(value)
  output.write_char('\n')
  written.val += encoded.length() + 1
}

///|
pub fn admit_paged_attention_parallel_promotion_evidence(
  candidate : PagedAttentionParallelCudaCandidate,
  trials : ArrayView[PagedAttentionParallelPromotionTrial],
  end_to_end : ArrayView[PagedAttentionParallelEndToEndTrial],
  limits : PagedAttentionParallelPromotionLimits,
) -> PagedAttentionParallelPromotionEvidence raise PagedAttentionParallelAotError {
  if trials.is_empty() ||
    trials.length() % limits.trials_per_shape != 0 ||
    trials.length() / limits.trials_per_shape > limits.maximum_shapes {
    raise PagedAttentionParallelMismatch(PromotionEvidence)
  }
  let policy = candidate.numerical_policy()
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
        raise PagedAttentionParallelMismatch(MicrobenchmarkWin)
      }
      if observation.shape_ordinal != shape + 1 ||
        observation.trial_ordinal != 0 {
        raise PagedAttentionParallelMismatch(ObservationOrder)
      }
      shape = observation.shape_ordinal
      trial = -1
      baseline_total = 0L
      candidate_total = 0L
    }
    if observation.trial_ordinal != trial + 1 ||
      observation.trial_ordinal >= limits.trials_per_shape {
      raise PagedAttentionParallelMismatch(ObservationOrder)
    }
    if observation.maximum_absolute_error_ppb >
      policy.maximum_absolute_error_ppb() ||
      observation.maximum_relative_error_ppb >
      policy.maximum_relative_error_ppb() ||
      (policy.requires_token_agreement() && !observation.token_agreement) {
      raise PagedAttentionParallelMismatch(Correctness)
    }
    if baseline_total > 9223372036854775807L - observation.baseline_duration_ns ||
      candidate_total > 9223372036854775807L - observation.candidate_duration_ns {
      raise PagedAttentionParallelOverflow(PromotionEvidence)
    }
    baseline_total += observation.baseline_duration_ns
    candidate_total += observation.candidate_duration_ns
    trial = observation.trial_ordinal
  }
  if trial != limits.trials_per_shape - 1 || candidate_total >= baseline_total {
    raise PagedAttentionParallelMismatch(MicrobenchmarkWin)
  }
  if end_to_end.length() != limits.end_to_end_trials {
    raise PagedAttentionParallelMismatch(PromotionEvidence)
  }
  for index, observation in end_to_end {
    if observation.trial_ordinal != index || !observation.token_agreement {
      raise PagedAttentionParallelMismatch(Correctness)
    }
    let allowed = observation.baseline_duration_ns +
      observation.baseline_duration_ns *
      limits.maximum_end_to_end_regression_ppm.to_int64() /
      1000000L
    if observation.candidate_duration_ns > allowed {
      raise PagedAttentionParallelMismatch(EndToEndRegression)
    }
  }
  let output = StringBuilder()
  let written = Ref(0)
  let append = value => {
    promotion_append(output, written, value, limits.maximum_canonical_bytes)
  }
  append("lunaflux.paged-attention-parallel-promotion-evidence.v1")
  append(candidate.candidate_digest().as_hex())
  append(candidate.reference_fallback().source_digest().as_hex())
  append(candidate.source_digest().as_hex())
  append(candidate.recipe_digest().as_hex())
  append(candidate.tile_program_digest().as_hex())
  append(candidate.tile_aot_digest().as_hex())
  append(candidate.numerical_policy().name())
  append(trials.length().to_string())
  for observation in trials {
    append(observation.shape_ordinal.to_string())
    append(observation.trial_ordinal.to_string())
    append(observation.baseline_duration_ns.to_string())
    append(observation.candidate_duration_ns.to_string())
    append(observation.maximum_absolute_error_ppb.to_string())
    append(observation.maximum_relative_error_ppb.to_string())
    append(observation.token_agreement.to_string())
    append(observation.correctness.as_hex())
    append(observation.dispatch_canary.as_hex())
  }
  append(end_to_end.length().to_string())
  for observation in end_to_end {
    append(observation.trial_ordinal.to_string())
    append(observation.baseline_duration_ns.to_string())
    append(observation.candidate_duration_ns.to_string())
    append(observation.token_agreement.to_string())
    append(observation.correctness.as_hex())
  }
  let canonical = @utf8.encode(output.to_string())
  {
    candidate_digest: candidate.candidate_digest(),
    reference_source_digest: candidate
    .reference_fallback()
    .source_digest()
    .as_hex(),
    trial_count: trials.length(),
    end_to_end_count: end_to_end.length(),
    canonical,
    digest: PagedAttentionParallelDigest(parallel_sha256(canonical)),
  }
}

///|
/// Candidate-only compiled binding. It cannot enter a runtime manifest and
/// carries no module bytes, loader, device, or execution authority.
pub struct PagedAttentionParallelReleaseCandidate {
  priv candidate : PagedAttentionParallelCudaCandidate
  priv entry_point : @catalog.AotKernelEntryPoint
  priv promotion_digest : PagedAttentionParallelDigest
  priv binding_record : Bytes
  priv binding_digest : PagedAttentionParallelDigest
}

///|
pub fn bind_promoted_paged_attention_parallel_release_candidate(
  candidate : PagedAttentionParallelCudaCandidate,
  entry_point : @catalog.AotKernelEntryPoint,
  compiled_source_digest : PagedAttentionParallelDigest,
  evidence : PagedAttentionParallelPromotionEvidence,
) -> PagedAttentionParallelReleaseCandidate raise PagedAttentionParallelAotError {
  if entry_point.entry_point_id() != candidate.entry_point_id() ||
    compiled_source_digest != candidate.source_digest() {
    raise PagedAttentionParallelMismatch(CompiledEntryPoint)
  }
  if evidence.candidate_digest != candidate.candidate_digest() ||
    evidence.reference_source_digest !=
    candidate.reference_fallback().source_digest().as_hex() {
    raise PagedAttentionParallelMismatch(PromotionEvidence)
  }
  let binding_record = @utf8.encode(
    "schema=lunaflux-paged-attention-parallel-release-candidate.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest().as_hex() +
    "\nsource_sha256=" +
    candidate.source_digest().as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\nmodule_sha256=" +
    entry_point.family().module_digest().sha256() +
    "\nentry_point_id=" +
    entry_point.entry_point_id().as_int().to_string() +
    "\npromotion_evidence_sha256=" +
    evidence.digest().as_hex() +
    "\nmanifest_bindable=false\n",
  )
  {
    candidate,
    entry_point,
    promotion_digest: evidence.digest(),
    binding_record,
    binding_digest: PagedAttentionParallelDigest(
      parallel_sha256(binding_record),
    ),
  }
}

///|
pub fn PagedAttentionParallelReleaseCandidate::manifest_bindable(
  _self : PagedAttentionParallelReleaseCandidate,
) -> Bool {
  false
}

///|
pub fn PagedAttentionParallelReleaseCandidate::candidate(
  self : PagedAttentionParallelReleaseCandidate,
) -> PagedAttentionParallelCudaCandidate {
  self.candidate
}

///|
pub fn PagedAttentionParallelReleaseCandidate::entry_point(
  self : PagedAttentionParallelReleaseCandidate,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn PagedAttentionParallelReleaseCandidate::promotion_digest(
  self : PagedAttentionParallelReleaseCandidate,
) -> PagedAttentionParallelDigest {
  self.promotion_digest
}

///|
pub fn PagedAttentionParallelReleaseCandidate::binding_record_bytes(
  self : PagedAttentionParallelReleaseCandidate,
) -> Bytes {
  self.binding_record
}

///|
pub fn PagedAttentionParallelReleaseCandidate::binding_digest(
  self : PagedAttentionParallelReleaseCandidate,
) -> PagedAttentionParallelDigest {
  self.binding_digest
}
