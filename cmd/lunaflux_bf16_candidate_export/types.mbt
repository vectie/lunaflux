///|
suberror LunaBf16CandidateExportCliError {
  InvalidArguments
  ModelAdmission
  ModelRootCleanup
  CandidatePreparation
  Publication
} derive(Debug, Eq)

///|
priv struct CandidateExportArguments {
  model_root : String
  config_locator : String
  config_sha256 : String
  model_content_sha256 : String
  toolchain_sha256 : String
  compiler_major : UInt32
  compiler_minor : UInt32
  compiler_patch : UInt32
  tokens_per_page : Int32
  total_page_count : Int32
  max_query_tokens : Int32
  max_page_table_entries : Int32
  absolute_new_output : String
}
