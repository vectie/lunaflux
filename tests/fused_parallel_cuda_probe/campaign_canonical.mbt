///|
fn render_fused_campaign_canonical(
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
  bindings : FusedProbeBindings,
  observation : FusedCampaignObservation,
  audit_record : Bytes,
  artifact_seal : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  evidence_seal : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> Bytes raise {
  let qkv = set.qkv_candidate()
  let residual = set.residual_norm_candidate()
  let output = StringBuilder()
  campaign_line(
    output, "schema", "lunaflux-fused-physical-qualification-evidence.v1",
  )
  campaign_line(output, "family_count", "2")
  campaign_line(
    output,
    "referee_shape_set_sha256",
    campaign_referee_shape_set_digest(),
  )
  campaign_line(output, "referee_shape_count", "4")
  campaign_line(
    output,
    "qkv_absolute_tolerance_ppb",
    qkv.numerical_policy().maximum_absolute_error_ppb().to_string(),
  )
  campaign_line(
    output,
    "qkv_relative_tolerance_ppb",
    qkv.numerical_policy().maximum_relative_error_ppb().to_string(),
  )
  campaign_line(
    output,
    "residual_absolute_tolerance_ppb",
    residual.numerical_policy().maximum_absolute_error_ppb().to_string(),
  )
  campaign_line(
    output,
    "residual_relative_tolerance_ppb",
    residual.numerical_policy().maximum_relative_error_ppb().to_string(),
  )
  campaign_line(output, "artifact_directory_sealed", "true")
  campaign_line(
    output,
    "artifact_directory_seal_sha256",
    artifact_seal.as_hex(),
  )
  campaign_line(output, "evidence_directory_sealed", "true")
  campaign_line(
    output,
    "evidence_directory_seal_sha256",
    evidence_seal.as_hex(),
  )
  for
    field in [
      "stderr_bytes", "contexts_live_after", "streams_live_after", "events_live_after",
      "graphs_live_after", "graph_execs_live_after", "allocations_live_after", "modules_live_after",
      "device_bytes_live_after", "pending_work_after",
    ] {
    campaign_line(output, field, "0")
  }
  campaign_line(output, "manifest_bindable", "false")
  campaign_line(output, "promotion_authority", "absent")
  let audit = @utf8.decode(audit_record) catch {
    _ => raise FusedCudaProbeRejected("physical audit record is not UTF-8")
  }
  require_fused_probe(
    audit.has_suffix("\n") && audit.split("\n").count() == 58,
    "physical audit record field count drifted",
  )
  output.write_string(audit)
  campaign_family(
    output,
    "qkv_",
    "qkv-positioned-rope-paged-kvwrite",
    qkv.candidate_digest(),
    qkv.source_digest(),
    qkv.recipe_digest(),
    qkv.compiler_policy(),
    qkv.target(),
    qkv.profile(),
    qkv.launch_dimensions(),
    qkv.function_symbol(),
    QkvPositionedRopePagedKvWriteAbiV1,
    bindings.qkv,
    qkv.numerical_policy(),
    qkv.dispatch_canary(),
    observation.qkv_absolute_error_ppb,
    observation.qkv_relative_error_ppb,
    observation.qkv_dispatch_canary,
  )
  campaign_family(
    output,
    "residual_",
    "residual-rmsnorm",
    residual.candidate_digest(),
    residual.source_digest(),
    residual.recipe_digest(),
    residual.compiler_policy(),
    residual.target(),
    residual.profile(),
    residual.launch_dimensions(),
    residual.function_symbol(),
    ResidualRmsNormAbiV1,
    bindings.residual,
    residual.numerical_policy(),
    residual.dispatch_canary(),
    observation.residual_absolute_error_ppb,
    observation.residual_relative_error_ppb,
    observation.residual_dispatch_canary,
  )
  @utf8.encode(output.to_string())
}
