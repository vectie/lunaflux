///|
fn load_candidate_projection_records(
  arguments : Qwen3CandidateExportArguments,
) -> ReadOnlyArray[@luna_projection_strategy.ProjectionAutotuneRecord] raise Qwen3Bf16CandidateExportCliError {
  @projection_tuning_file.load_projection_tuning_records(
    arguments.projection_tuning,
    backend=@luna_projection_strategy.ProjectionBackendId::new("cuda-aot-v1"),
    architecture=@luna_projection_strategy.ProjectionTargetId::new("sm_120"),
    toolchain_sha256=arguments.toolchain_sha256,
  ) catch {
    _ => raise CandidatePreparation
  }
}
