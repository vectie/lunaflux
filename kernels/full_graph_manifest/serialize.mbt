///|
fn append_manifest_dimensions(
  output : StringBuilder,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Unit {
  output.write_string(
    "{\"grid_x\":" +
    dimensions.grid_x().to_string() +
    ",\"grid_y\":" +
    dimensions.grid_y().to_string() +
    ",\"grid_z\":" +
    dimensions.grid_z().to_string() +
    ",\"block_x\":" +
    dimensions.block_x().to_string() +
    ",\"block_y\":" +
    dimensions.block_y().to_string() +
    ",\"block_z\":" +
    dimensions.block_z().to_string() +
    ",\"shared_memory_bytes\":" +
    dimensions.shared_memory_bytes().to_string() +
    "}",
  )
}

///|
fn append_manifest_modules(
  output : StringBuilder,
  modules : ArrayView[FullGraphManifestModule],
) -> Int {
  let mut entry_count = 0
  output.write_string("\"modules\":[")
  for module_index, item in modules {
    if module_index > 0 {
      output.write_string(",")
    }
    output.write_string(
      "{\"sha256\":\"" +
      item.digest.sha256() +
      "\",\"path\":\"" +
      item.path +
      "\",\"entry_points\":[",
    )
    for entry_index, entry in item.entries {
      if entry_index > 0 {
        output.write_string(",")
      }
      output.write_string(
        "{\"family_id\":" +
        entry.family_id.as_int().to_string() +
        ",\"entry_point_id\":" +
        entry.entry_point_id.as_int().to_string() +
        ",\"symbol\":\"" +
        entry.function_symbol +
        "\"}",
      )
      entry_count += 1
    }
    output.write_string("]}")
  }
  output.write_string("],")
  entry_count
}

///|
fn append_manifest_operations(
  output : StringBuilder,
  artifacts : ArrayView[FullGraphAotOperationArtifact],
  modules : ArrayView[FullGraphManifestModule],
) -> Unit raise FullGraphManifestError {
  output.write_string("\"operations\":[")
  for index, artifact in artifacts {
    if index > 0 {
      output.write_string(",")
    }
    let family = artifact.entry_point.family()
    guard manifest_module_index(modules, family.module_digest())
      is Some(module_index) else {
      raise Mismatch(rule=ModuleOrdering, operation_index=index)
    }
    output.write_string(
      "{\"operation_id\":" +
      artifact.operation_id.as_int().to_string() +
      ",\"module_index\":" +
      module_index.to_string() +
      ",\"family_id\":" +
      family.family_id().as_int().to_string() +
      ",\"entry_point_id\":" +
      artifact.entry_point.entry_point_id().as_int().to_string() +
      ",\"workspace_bytes\":" +
      artifact.workspace_bytes.to_string() +
      ",\"workspace_alignment\":" +
      artifact.workspace_alignment.to_string() +
      ",\"dimensions\":",
    )
    append_manifest_dimensions(output, artifact.dimensions)
    output.write_string("}")
  }
  output.write_string("]}")
}

///|
fn serialize_full_graph_manifest(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : ArrayView[FullGraphAotOperationArtifact],
  modules : Array[FullGraphManifestModule],
  limits : FullGraphManifestLimits,
) -> FullGraphExecutionManifestInput raise FullGraphManifestError {
  let profile = contracts.profiles()[0]
  let identity = model_plan.identity()
  let output = StringBuilder()
  output.write_string(
    "{\"schema_version\":2,\"model_identity\":{" +
    "\"content_sha256\":\"" +
    identity.content().as_hex() +
    "\",\"plan_sha256\":\"" +
    identity.plan().as_hex() +
    "\"},\"device_target\":{" +
    "\"compute_major\":" +
    target.compute_major().to_string() +
    ",\"compute_minor\":" +
    target.compute_minor().to_string() +
    ",\"supports_bf16\":" +
    (if target.supports_bf16() { "true" } else { "false" }) +
    ",\"supports_cublas_lt\":" +
    (if target.supports_cublas_lt() { "true" } else { "false" }) +
    "},\"catalog_version\":3,\"profile\":{\"id\":" +
    profile.id().as_int().to_string() +
    "},",
  )
  let entry_count = append_manifest_modules(output, modules)
  append_manifest_operations(output, artifacts, modules)
  let bytes = @utf8.encode(output.to_string())
  if bytes.length().to_int64() > limits.max_manifest_bytes {
    raise LimitExceeded(rule=ManifestBytes, maximum=limits.max_manifest_bytes)
  }
  {
    bytes,
    digest_sha256: @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    module_count: modules.length(),
    entry_point_count: entry_count,
    operation_count: artifacts.length(),
  }
}
