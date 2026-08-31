///|
suberror Qwen3Bf16CandidateExportCliError {
  InvalidArguments
  ModelAdmission
  ModelRootCleanup
  CandidatePreparation
  ResidualFusionPreparation
  ResidualProductionPreparation
  QwenKvWritePreparation
  QwenIngressPreparation
  QwenReadOnlyPreparation
  QwenFusionMissing
  Publication
} derive(Debug, Eq)

///|
fn candidate_export_error_name(
  error : Qwen3Bf16CandidateExportCliError,
) -> String {
  match error {
    InvalidArguments => "invalid-arguments"
    ModelAdmission => "model-admission"
    ModelRootCleanup => "model-root-cleanup"
    CandidatePreparation => "candidate-preparation"
    ResidualFusionPreparation => "residual-fusion-preparation"
    ResidualProductionPreparation => "residual-production-preparation"
    QwenKvWritePreparation => "qwen-kv-write-preparation"
    QwenIngressPreparation => "qwen-ingress-preparation"
    QwenReadOnlyPreparation => "qwen-readonly-preparation"
    QwenFusionMissing => "qwen-fusion-missing"
    Publication => "publication"
  }
}

///|
fn fused_parallel_rule_name(
  rule : @luna_cuda_fused_parallel_aot.FusedParallelAotRule,
) -> String {
  match rule {
    Limits => "limits"
    Plan => "plan"
    Operation => "operation"
    OperationOrder => "operation-order"
    Shape => "shape"
    Profile => "profile"
    KvLayout => "kv-layout"
    DeviceTarget => "device-target"
    CompilerPolicy => "compiler-policy"
    ReferenceFallback => "reference-fallback"
    OperandOrder => "operand-order"
    OperandSize => "operand-size"
    OperandAlignment => "operand-alignment"
    StaticResources => "static-resources"
    Source => "source"
    Recipe => "recipe"
    Digest => "digest"
    CompileReceipt => "compile-receipt"
    CompiledArtifact => "compiled-artifact"
    Determinism => "determinism"
    KernelAbi => "kernel-abi"
    Evidence => "evidence"
    ObservationOrder => "observation-order"
    Correctness => "correctness"
    MicrobenchmarkWin => "microbenchmark-win"
  }
}

///|
fn fused_parallel_error_name(
  error : @luna_cuda_fused_parallel_aot.FusedParallelAotError,
) -> String {
  match error {
    FusedParallelInvalid(rule) => "invalid-" + fused_parallel_rule_name(rule)
    FusedParallelUnsupported(rule) =>
      "unsupported-" + fused_parallel_rule_name(rule)
    FusedParallelMismatch(rule) => "mismatch-" + fused_parallel_rule_name(rule)
    FusedParallelLimitExceeded(rule) =>
      "limit-exceeded-" + fused_parallel_rule_name(rule)
    FusedParallelOverflow(rule) => "overflow-" + fused_parallel_rule_name(rule)
  }
}

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
  max_query_rows : Int
  max_query_tokens : Int
  max_page_table_entries : Int
  absolute_new_output : String
}
