///|
suberror Qwen3Bf16ReleaseBindError {
  InvalidArguments
  ModelAdmission
  CompiledSetAdmission(String)
  KernelBinding
  BootstrapAdmission(@execution_manifest_file.ExecutionManifestAdmissionStage)
  Publication
} derive(Debug, Eq)

///|
struct Qwen3ReleaseBindArguments {
  compute_major : Int32
  compute_minor : Int32
  max_activation_arena_bytes : Int64
  model_root : String
  config_locator : String
  config_sha256 : String
  model_content_sha256 : String
  numeric_weights_locator : String
  numeric_weight_artifact_sha256 : String
  weight_route_manifest_sha256 : String
  compiled_root : String
  toolchain_manifest : String
  toolchain_sha256 : String
  compiler_major : UInt32
  compiler_minor : UInt32
  compiler_patch : UInt32
  tokens_per_page : Int32
  total_page_count : Int32
  max_query_rows : Int32
  max_query_tokens : Int32
  max_page_table_entries : Int32
  absolute_new_output : String
  projection_tuning : @projection_tuning_file.ProjectionTuningInput?
  projection_folds : (String, String, String)?
  compiler_register_limit : Int32
}

///|
struct Qwen3ReleaseBindInputs {
  plan : @plan.ModelPlan
  weights : @numeric_weight_file.NumericWeightFileInspection
}

///|
struct Qwen3ReleaseBindEvidence {
  planned_activation_arena_bytes : Int64
  model_plan_sha256 : String
  manifest_sha256 : String
  bootstrap_sha256 : String
  bootstrap_source_sha256 : String
  inventory_sha256 : String
  plan_sha256 : String
}
