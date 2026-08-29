///|
pub fn evaluate_positioned_paged_kv_write_fake_campaign(
  candidate : PagedKvWriteCudaCandidate,
  observations : ArrayView[PagedKvWriteFakeCampaignObservation],
) -> PagedKvWriteFakeCampaignReport raise PagedKvWriteAotError {
  if observations.length() < 4 {
    raise Invalid(Fallback)
  }
  let output = StringBuilder()
  output.write_string(
    "schema=lunaflux-positioned-paged-kv-write-fake-campaign.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest.as_hex() +
    "\n",
  )
  for index, observation in observations {
    let expected = paged_kv_write_checked_mul(
      observation.live_tokens.to_int64(),
      observation.kv_width.to_int64(),
      Fallback,
    )
    if observation.kv_width != candidate.kv_width ||
      observation.key_writes != expected ||
      observation.value_writes != expected ||
      observation.observed_untouched != observation.expected_untouched {
      raise Mismatch(Fallback)
    }
    output.write_string(
      "observation_" +
      index.to_string() +
      "=" +
      observation.live_tokens.to_string() +
      "," +
      observation.kv_width.to_string() +
      "," +
      expected.to_string() +
      "," +
      observation.expected_untouched.to_string() +
      "\n",
    )
  }
  output.write_string(
    "exact_key_mutations=true\nexact_value_mutations=true\n" +
    "untouched_sentinels=true\nguard_silent_return_rejected=true\n" +
    "physical_evidence=false\nmanifest_bindable=false\n" +
    "promotion_authority=false\n",
  )
  let canonical = @utf8.encode(output.to_string())
  {
    candidate: candidate.candidate_digest,
    observation_count: observations.length(),
    canonical,
    digest: PagedKvWriteDigest(paged_kv_write_sha256(canonical)),
  }
}

///|
pub fn PagedKvWriteFakeCampaignReport::candidate_digest(
  self : PagedKvWriteFakeCampaignReport,
) -> PagedKvWriteDigest {
  self.candidate
}

///|
pub fn PagedKvWriteFakeCampaignReport::observation_count(
  self : PagedKvWriteFakeCampaignReport,
) -> Int {
  self.observation_count
}

///|
pub fn PagedKvWriteFakeCampaignReport::canonical_bytes(
  self : PagedKvWriteFakeCampaignReport,
) -> Bytes {
  self.canonical
}

///|
pub fn PagedKvWriteFakeCampaignReport::digest(
  self : PagedKvWriteFakeCampaignReport,
) -> PagedKvWriteDigest {
  self.digest
}

///|
pub fn PagedKvWriteFakeCampaignReport::physical_evidence(
  _self : PagedKvWriteFakeCampaignReport,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteFakeCampaignReport::manifest_bindable(
  _self : PagedKvWriteFakeCampaignReport,
) -> Bool {
  false
}

///|
pub fn PagedKvWriteFakeCampaignReport::promotion_authority(
  _self : PagedKvWriteFakeCampaignReport,
) -> Bool {
  false
}
