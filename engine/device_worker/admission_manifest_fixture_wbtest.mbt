///|
fn admission_manifest(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  module_bytes : BytesView,
) -> String {
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes))
  let kinds : Array[@plan.OperationKind] = []
  let mut entries = ""
  let mut operations = ""
  for index, operation in plan.operations() {
    if index > 0 {
      operations += ","
    }
    let mut kind_index = 0
    for candidate_index, candidate in kinds {
      if candidate == operation.kind() {
        kind_index = candidate_index + 1
      }
    }
    if kind_index == 0 {
      kinds.push(operation.kind())
      kind_index = kinds.length()
      if kind_index > 1 {
        entries += ","
      }
      entries += "{\"family_id\":" +
        kind_index.to_string() +
        ",\"entry_point_id\":" +
        kind_index.to_string() +
        ",\"symbol\":\"kernel_" +
        kind_index.to_string() +
        "\"}"
    }
    let workspace = operation.workspace_bytes()
    operations += "{\"operation_id\":" +
      operation.id().as_int().to_string() +
      ",\"module_index\":0,\"family_id\":" +
      kind_index.to_string() +
      ",\"entry_point_id\":" +
      kind_index.to_string() +
      ",\"workspace_bytes\":" +
      workspace.to_string() +
      ",\"workspace_alignment\":" +
      (if workspace > 0L { "16" } else { "1" }) +
      ",\"dimensions\":{\"grid_x\":1,\"grid_y\":1,\"grid_z\":1," +
      "\"block_x\":32,\"block_y\":1,\"block_z\":1," +
      "\"shared_memory_bytes\":0}}"
  }
  "{\"schema_version\":2,\"model_identity\":{\"content_sha256\":\"" +
  plan.identity().content().as_hex() +
  "\",\"plan_sha256\":\"" +
  plan.identity().plan().as_hex() +
  "\"},\"device_target\":{\"compute_major\":" +
  target.compute_major().to_string() +
  ",\"compute_minor\":" +
  target.compute_minor().to_string() +
  ",\"supports_bf16\":true,\"supports_cublas_lt\":false}," +
  "\"catalog_version\":3,\"profile\":{\"id\":1},\"modules\":[{" +
  "\"sha256\":\"" +
  digest +
  "\",\"path\":\"kernel.cubin\",\"entry_points\":[" +
  entries +
  "]}],\"operations\":[" +
  operations +
  "]}"
}
