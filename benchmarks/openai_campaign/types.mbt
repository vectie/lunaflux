///|
pub(all) suberror OpenAIResponsesCampaignError {
  CampaignInvalidDigest
  CampaignInvalidJson
  CampaignInvalidSchema
  CampaignInvalidDeclaration
  CampaignInvalidEndpoint
  CampaignInvalidCredentialBoundary
  CampaignInvalidShape
  CampaignInvalidCapture
  CampaignInvalidTimestamp
  CampaignInvalidHex
  CampaignAdapterRejected
} derive(Debug, Eq)

///|
priv struct CampaignEngineDeclaration {
  engine : @evidence.BenchmarkEngine
  endpoint : String
  revision : @evidence.BenchmarkDigest
  image : @evidence.BenchmarkDigest
  configuration : @evidence.BenchmarkDigest
  executable : @evidence.BenchmarkDigest
  identity : @openai_adapter.OpenAIBaselineIdentity
}

///|
priv struct CampaignProfileDeclaration {
  profile : @evidence.BenchmarkProfile
  workload : @evidence.BenchmarkDigest
  request_input_tokens : FixedArray[Int64]
}

///|
pub struct OpenAIResponsesCampaignDeclaration {
  priv digest : @evidence.BenchmarkDigest
  priv pins : @openai_adapter.OpenAIComparisonEnvironmentPins
  priv tokenizer : @evidence.BenchmarkDigest
  priv corpus : @evidence.BenchmarkDigest
  priv protocol : @evidence.BenchmarkDigest
  priv engines : FixedArray[CampaignEngineDeclaration]
  priv profiles : FixedArray[CampaignProfileDeclaration]
}

///|
pub struct OpenAIResponsesCampaignReplayEvidence {
  priv declaration : @evidence.BenchmarkDigest
  priv raw_capture_set : @evidence.BenchmarkDigest
  priv measurements : @openai_adapter.OpenAIResponsesComparisonMeasurementSet
  priv canonical : Bytes
  priv digest : @evidence.BenchmarkDigest
}
