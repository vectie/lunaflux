///|
/// Summarizes the existing production paged-execution admission. This is not
/// the Phase-5 LunaTile capability-manifest format and does not manufacture a
/// signature receipt, graph execution, device context, or readiness claim.
pub fn render_paged(
  admission : @execution_manifest_file.PagedExecutionAdmission,
) -> String {
  let blueprint = admission.blueprint()
  let identity = blueprint.identity()
  let target = blueprint.target()
  let layout = admission.layout()
  let step_limits = admission.device_step_limits()
  let artifacts = admission.artifacts()
  let output = StringBuilder()
  append_line(output, "LunaFlux admitted paged AOT execution")
  append_line(output, "model content sha256: \{identity.content().as_hex()}")
  append_line(output, "model plan sha256: \{identity.plan().as_hex()}")
  append_line(
    output,
    "target: sm_\{target.compute_major()}\{target.compute_minor()}",
  )
  append_line(output, "target.bf16: \{target.supports_bf16()}")
  append_line(output, "target.cublas_lt: \{target.supports_cublas_lt()}")
  append_line(
    output,
    "catalog version: \{blueprint.catalog_version().as_int()}",
  )
  append_line(output, "profile id: \{admission.profile_id().as_int()}")
  append_line(output, "operation count: \{blueprint.steps().length()}")
  append_line(output, "module count: \{artifacts.module_count()}")
  append_line(output, "entry point count: \{artifacts.entry_point_count()}")
  append_line(output, "module bytes: \{artifacts.total_module_bytes()}")
  append_line(output, "max rows: \{step_limits.max_rows()}")
  append_line(output, "max step tokens: \{step_limits.max_tokens()}")
  append_line(output, "max page-table cells: \{step_limits.max_pages()}")
  append_line(output, "tokens per page: \{layout.tokens_per_page()}")
  append_line(output, "total pages: \{layout.total_page_count()}")
  append_line(output, "KV arena bytes: \{layout.arena_bytes()}")
  append_line(
    output,
    "activation/workspace arena bytes: \{blueprint.activation_workspace_arena_bytes()}",
  )
  append_line(output, "execution disposition: admitted_paged_aot")
  append_line(output, "device readiness: not claimed")
  output.to_string()
}
