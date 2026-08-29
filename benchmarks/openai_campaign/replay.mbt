///|
/// Replays exactly 81 authenticated raw capture documents in canonical
/// profile/trial/engine order. Raw documents are consumed without filesystem
/// mutation. The returned type carries measurements only and has no correctness
/// or comparison-pass authority.
pub fn replay_openai_responses_campaign(
  declaration : OpenAIResponsesCampaignDeclaration,
  captures : ArrayView[Bytes],
) -> OpenAIResponsesCampaignReplayEvidence raise OpenAIResponsesCampaignError {
  if captures.length() != 81 {
    raise CampaignInvalidShape
  }
  let trials : Array[@openai_adapter.OpenAIResponsesTrialEvidence] = []
  let capture_record = StringBuilder()
  capture_record.write_string(
    "schema=lunaflux-openai-responses-capture-set.v1\ndeclaration_sha256=" +
    declaration.digest.as_hex() +
    "\n",
  )
  for index, bytes in captures {
    let (trial, raw_digest) = replay_campaign_trial(declaration, index, bytes)
    trials.push(trial)
    capture_record.write_string(
      "trial_" + index.to_string() + "_sha256=" + raw_digest.as_hex() + "\n",
    )
  }
  let raw_capture_set = campaign_sha256(
    @utf8.encode(capture_record.to_string()),
  )
  let measurements = @openai_adapter.admit_openai_responses_comparison_measurements(
    declaration.pins,
    trials,
  ) catch {
    _ => raise CampaignAdapterRejected
  }
  measurements.require_responses_v1_observation()
  let canonical = @utf8.encode(
    "schema=lunaflux-openai-responses-campaign-replay.v1\n" +
    "declaration_sha256=" +
    declaration.digest.as_hex() +
    "\nraw_capture_set_sha256=" +
    raw_capture_set.as_hex() +
    "\ntrial_count=81\nobservation_protocol=openai.responses.sse.v1\n" +
    "timestamp_authority=external_live_runner\n" +
    "network_authority=external_live_runner\n" +
    "credential_authority=external_live_runner\n" +
    "replay_authority=offline_admission_only\n" +
    "raw_artifact_mutation=none\ncomparison_authority=none\n" +
    "correctness_authority=none\n",
  )
  {
    declaration: declaration.digest,
    raw_capture_set,
    measurements,
    canonical,
    digest: campaign_sha256(canonical),
  }
}

///|
pub fn OpenAIResponsesCampaignReplayEvidence::declaration_digest(
  self : OpenAIResponsesCampaignReplayEvidence,
) -> @evidence.BenchmarkDigest {
  self.declaration
}

///|
pub fn OpenAIResponsesCampaignReplayEvidence::raw_capture_set_digest(
  self : OpenAIResponsesCampaignReplayEvidence,
) -> @evidence.BenchmarkDigest {
  self.raw_capture_set
}

///|
pub fn OpenAIResponsesCampaignReplayEvidence::measurements(
  self : OpenAIResponsesCampaignReplayEvidence,
) -> @openai_adapter.OpenAIResponsesComparisonMeasurementSet {
  self.measurements
}

///|
pub fn OpenAIResponsesCampaignReplayEvidence::canonical_bytes(
  self : OpenAIResponsesCampaignReplayEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn OpenAIResponsesCampaignReplayEvidence::digest(
  self : OpenAIResponsesCampaignReplayEvidence,
) -> @evidence.BenchmarkDigest {
  self.digest
}
