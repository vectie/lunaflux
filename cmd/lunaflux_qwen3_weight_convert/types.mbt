///|
priv suberror Qwen3WeightConvertError {
  InvalidArguments
  ModelRoot
  ConfigAdmission
  SourceAdmission
  ConversionPlan
  Publication(status~ : Int)
  OutputAdmission
  Cleanup
}

///|
priv struct Qwen3WeightConvertArguments {
  model_root : String
  config_locator : String
  config_sha256 : String
  model_content_sha256 : String
  source_locator : String
  source_sha256 : String
  max_batch_rows : Int
  output_locator : String
}

///|
priv struct Qwen3NumericConversionPlan {
  header : Bytes
  source_regions : Bytes
  payload_bytes : Int64
}

///|
priv struct Qwen3WeightConvertReport {
  source_artifact_sha256 : String
  source_manifest_sha256 : String
  output_artifact_sha256 : String
  route_manifest_sha256 : String
}
