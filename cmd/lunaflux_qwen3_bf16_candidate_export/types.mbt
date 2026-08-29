///|
suberror Qwen3Bf16CandidateExportCliError {
  InvalidArguments
  ModelAdmission
  ModelRootCleanup
  CandidatePreparation
  Publication
} derive(Debug, Eq)

///|
priv struct Qwen3CandidateExportArguments {
  model_root : String
  config_locator : String
  config_sha256 : String
  model_content_sha256 : String
  numeric_weights_locator : String
  numeric_weight_artifact_sha256 : String
  weight_route_manifest_sha256 : String
  toolchain_sha256 : String
  compiler_major : UInt
  compiler_minor : UInt
  compiler_patch : UInt
  tokens_per_page : Int
  total_page_count : Int
  max_query_tokens : Int
  max_page_table_entries : Int
  absolute_new_output : String
}
