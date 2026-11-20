///|
fn print_qwen3_export_summary(
  arguments : Qwen3CandidateExportArguments,
  evidence : @luna_bf16_candidate_export.LunaBf16CandidateExportEvidence,
) -> Unit {
  println("schema=lunaflux-qwen3-bf16-candidate-export-cli.v1")
  println("recipe=dense_qwen3_bf16_paged_aot_v12")
  println(
    "weight_route_manifest_sha256=" + arguments.weight_route_manifest_sha256,
  )
  println("model_content_sha256=" + evidence.model_content_sha256())
  println("model_plan_sha256=" + evidence.model_plan_sha256())
  println("target=" + evidence.target())
  println("max_query_rows=" + arguments.max_query_rows.to_string())
  println("max_query_tokens=" + arguments.max_query_tokens.to_string())
  println("operation_count=" + evidence.operation_count().to_string())
  println("candidate_root=" + arguments.absolute_new_output + "/candidate-root")
  println(
    "candidate_inventory=" +
    arguments.absolute_new_output +
    "/candidate.files.sha256#sha256=" +
    evidence.inventory_sha256(),
  )
  println("candidate_set_sha256=" + evidence.candidate_set_sha256())
  println("export_record_sha256=" + evidence.export_record_sha256())
  println("compiler_invoked=0")
  println("device_opened=0")
  println("runtime_authority=0")
  println("physical_readiness=0")
}
