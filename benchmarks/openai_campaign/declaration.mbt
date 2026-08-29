///|
fn parse_campaign_engine_declaration(
  value : Json,
  expected_index : Int,
) -> CampaignEngineDeclaration raise OpenAIResponsesCampaignError {
  let object = campaign_require_object(value, 6)
  let engine = campaign_engine(campaign_string(object, "engine"))
  if engine != campaign_engine_at(expected_index) {
    raise CampaignInvalidShape
  }
  let endpoint = campaign_string(object, "endpoint")
  campaign_loopback_endpoint(endpoint)
  let revision = campaign_digest(campaign_string(object, "revision_sha256"))
  let image = campaign_digest(campaign_string(object, "image_sha256"))
  let configuration = campaign_digest(
    campaign_string(object, "configuration_sha256"),
  )
  let executable = campaign_digest(campaign_string(object, "executable_sha256"))
  let identity = @openai_adapter.OpenAIBaselineIdentity::new(
    engine, revision, image, configuration,
  )
  { engine, endpoint, revision, image, configuration, executable, identity }
}

///|
fn parse_campaign_profile_declaration(
  value : Json,
  expected_index : Int,
) -> CampaignProfileDeclaration raise OpenAIResponsesCampaignError {
  let object = campaign_require_object(value, 3)
  let profile = campaign_profile(campaign_string(object, "profile"))
  if profile != campaign_profile_at(expected_index) {
    raise CampaignInvalidShape
  }
  let raw_tokens = campaign_array(object, "request_input_tokens")
  if raw_tokens.is_empty() ||
    raw_tokens.length() > @evidence.maximum_request_events_per_summary() {
    raise CampaignInvalidShape
  }
  let tokens = FixedArray::makei(raw_tokens.length(), index => {
    guard raw_tokens[index] is String(value) else { raise CampaignInvalidJson }
    campaign_decimal_i64(value, true)
  })
  {
    profile,
    workload: campaign_digest(campaign_string(object, "workload_sha256")),
    request_input_tokens: tokens,
  }
}

///|
/// Parses and authenticates one immutable campaign declaration. The expected
/// SHA-256 is supplied separately by the operator; no credential enters this
/// API. Endpoints must be exact IPv4 loopback addresses and the only admitted
/// credential boundary is an external inherited-descriptor live runner.
pub fn OpenAIResponsesCampaignDeclaration::parse(
  bytes : BytesView,
  expected_sha256 : String,
) -> OpenAIResponsesCampaignDeclaration raise OpenAIResponsesCampaignError {
  let expected = campaign_digest(expected_sha256)
  let actual = campaign_sha256(bytes)
  if actual != expected {
    raise CampaignInvalidDigest
  }
  let root = campaign_require_object(campaign_parse_json(bytes, 8), 13)
  if campaign_string(root, "schema") !=
    "lunaflux-openai-responses-campaign-declaration.v1" {
    raise CampaignInvalidSchema
  }
  if campaign_string(root, "credential_ingress") !=
    "external.inherited-descriptor.v1" {
    raise CampaignInvalidCredentialBoundary
  }
  ignore(campaign_digest(campaign_string(root, "credential_policy_sha256")))
  let raw_engines = campaign_array(root, "engines")
  let raw_profiles = campaign_array(root, "profiles")
  if raw_engines.length() != 3 || raw_profiles.length() != 9 {
    raise CampaignInvalidShape
  }
  let engines = FixedArray::makei(3, index => {
    parse_campaign_engine_declaration(raw_engines[index], index)
  })
  if engines[0].endpoint == engines[1].endpoint ||
    engines[0].endpoint == engines[2].endpoint ||
    engines[1].endpoint == engines[2].endpoint {
    raise CampaignInvalidEndpoint
  }
  let profiles = FixedArray::makei(9, index => {
    parse_campaign_profile_declaration(raw_profiles[index], index)
  })
  {
    digest: actual,
    pins: @openai_adapter.OpenAIComparisonEnvironmentPins::new(
      campaign_digest(campaign_string(root, "hardware_sha256")),
      campaign_digest(campaign_string(root, "driver_sha256")),
      campaign_digest(campaign_string(root, "toolkit_sha256")),
      campaign_digest(campaign_string(root, "model_sha256")),
      campaign_digest(campaign_string(root, "randomization_sha256")),
    ),
    tokenizer: campaign_digest(campaign_string(root, "tokenizer_sha256")),
    corpus: campaign_digest(campaign_string(root, "corpus_sha256")),
    protocol: campaign_digest(campaign_string(root, "protocol_sha256")),
    engines,
    profiles,
  }
}

///|
pub fn OpenAIResponsesCampaignDeclaration::digest(
  self : OpenAIResponsesCampaignDeclaration,
) -> @evidence.BenchmarkDigest {
  self.digest
}

///|
pub fn OpenAIResponsesCampaignDeclaration::endpoint(
  self : OpenAIResponsesCampaignDeclaration,
  index : Int,
) -> String raise OpenAIResponsesCampaignError {
  if index < 0 || index >= 3 {
    raise CampaignInvalidShape
  }
  self.engines[index].endpoint
}

///|
pub fn OpenAIResponsesCampaignDeclaration::engine_revision(
  self : OpenAIResponsesCampaignDeclaration,
  index : Int,
) -> @evidence.BenchmarkDigest raise OpenAIResponsesCampaignError {
  if index < 0 || index >= 3 {
    raise CampaignInvalidShape
  }
  self.engines[index].revision
}

///|
pub fn OpenAIResponsesCampaignDeclaration::engine_image(
  self : OpenAIResponsesCampaignDeclaration,
  index : Int,
) -> @evidence.BenchmarkDigest raise OpenAIResponsesCampaignError {
  if index < 0 || index >= 3 {
    raise CampaignInvalidShape
  }
  self.engines[index].image
}

///|
pub fn OpenAIResponsesCampaignDeclaration::engine_configuration(
  self : OpenAIResponsesCampaignDeclaration,
  index : Int,
) -> @evidence.BenchmarkDigest raise OpenAIResponsesCampaignError {
  if index < 0 || index >= 3 {
    raise CampaignInvalidShape
  }
  self.engines[index].configuration
}

///|
pub fn OpenAIResponsesCampaignDeclaration::engine_executable(
  self : OpenAIResponsesCampaignDeclaration,
  index : Int,
) -> @evidence.BenchmarkDigest raise OpenAIResponsesCampaignError {
  if index < 0 || index >= 3 {
    raise CampaignInvalidShape
  }
  self.engines[index].executable
}
