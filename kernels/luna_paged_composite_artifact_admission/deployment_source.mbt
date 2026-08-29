///|
const FUSED_PRODUCTION_SOURCE_SCHEMA : String = "lunaflux-fused-production-runtime-source.v1"

///|
fn deployment_line(out : StringBuilder, key : String, value : String) -> Unit {
  out.write_string(key + "=" + value + "\n")
}

///|
fn fused_production_source_canonical(
  artifacts : ApprovedFusedQkvReadOnlyProductionArtifacts,
) -> Bytes {
  let ro = artifacts.readonly_identity
  let kv = artifacts.kv_write
  let launch = ro.launch
  let profile = kv.profile
  let fused = @luna_fused_artifact_admission.export_digest_pinned_fused_runtime_source(
    artifacts.fused_catalog,
    artifacts.fused_bundle,
  ) catch {
    _ => abort("approved fused production aggregate lost its fused source")
  }
  let out = StringBuilder()
  deployment_line(out, "schema", FUSED_PRODUCTION_SOURCE_SCHEMA)
  deployment_line(out, "fused_source_hex", @crypto.bytes_to_hex_string(fused))
  for
    item in [
      ("readonly_manifest_sha256", ro.manifest),
      ("readonly_catalog_sha256", ro.catalog),
      ("readonly_candidate_sha256", ro.candidate),
      ("readonly_source_sha256", ro.source),
      ("readonly_recipe_sha256", ro.recipe),
      ("readonly_fallback_source_sha256", ro.fallback_source),
      ("readonly_fallback_recipe_sha256", ro.fallback_recipe),
      ("readonly_binding_sha256", ro.compiled_binding),
      ("readonly_module_sha256", ro.module_digest),
      ("readonly_physical_sha256", ro.physical_evidence),
      ("readonly_policy_sha256", ro.approved_policy),
      ("readonly_toolchain_sha256", ro.toolchain),
      ("readonly_driver_sha256", ro.driver),
      ("readonly_runtime_device_sha256", ro.runtime_device),
      ("kv_model_content_sha256", kv.model_content),
      ("kv_model_plan_sha256", kv.model_plan),
      ("kv_candidate_sha256", kv.candidate),
      ("kv_source_sha256", kv.source),
      ("kv_recipe_sha256", kv.recipe),
      ("kv_binding_sha256", kv.compiled_binding),
      ("kv_module_sha256", kv.module_digest),
      ("kv_physical_sha256", kv.physical_evidence),
      ("kv_policy_sha256", kv.approved_policy),
      ("kv_fallback_source_sha256", kv.fallback_source),
      ("kv_fallback_recipe_sha256", kv.fallback_recipe),
      ("kv_toolchain_sha256", kv.toolchain),
      ("kv_driver_sha256", kv.driver),
    ] {
    deployment_line(out, item.0, item.1)
  }
  for
    item in [
      ("readonly_operation", ro.operation),
      ("readonly_activation", ro.activation),
      ("readonly_row_width", ro.input_row_width),
      ("readonly_profile_id", ro.profile_id),
      ("readonly_profile_rows", ro.profile_rows),
      ("readonly_profile_tokens", ro.profile_tokens),
      ("readonly_profile_pages", ro.profile_page_entries),
      ("readonly_tokens_per_page", ro.tokens_per_page),
      ("readonly_grid_x", launch.grid_x()),
      ("readonly_grid_y", launch.grid_y()),
      ("readonly_grid_z", launch.grid_z()),
      ("readonly_block_x", launch.block_x()),
      ("readonly_block_y", launch.block_y()),
      ("readonly_block_z", launch.block_z()),
      ("readonly_shared_bytes", launch.shared_memory_bytes()),
      ("readonly_canary", ro.dispatch_canary),
      ("readonly_canary_cells", ro.dispatch_canary_cells),
      ("kv_operation", kv.operation.as_int()),
      ("kv_activation", kv.activation.as_int()),
      ("kv_row_width", kv.row_width),
      ("kv_profile_id", profile.id().as_int()),
      ("kv_profile_rows", profile.max_query_rows()),
      ("kv_profile_tokens", profile.max_query_tokens()),
      ("kv_profile_pages", profile.max_page_table_entries()),
      ("kv_tokens_per_page", kv.tokens_per_page),
      ("kv_total_page_count", kv.total_page_count),
    ] {
    deployment_line(out, item.0, item.1.to_string())
  }
  deployment_line(out, "readonly_page_stride", ro.page_stride_bytes.to_string())
  deployment_line(
    out,
    "readonly_page_capacity",
    ro.physical_page_capacity.to_string(),
  )
  deployment_line(
    out,
    "readonly_symbol_hex",
    @crypto.bytes_to_hex_string(@utf8.encode(ro.symbol)),
  )
  deployment_line(
    out,
    "readonly_module_hex",
    @crypto.bytes_to_hex_string(ro.module_bytes),
  )
  deployment_line(
    out,
    "readonly_device_uuid_hex",
    @crypto.bytes_to_hex_string(@utf8.encode(ro.device_uuid)),
  )
  deployment_line(
    out,
    "readonly_device_pci_hex",
    @crypto.bytes_to_hex_string(@utf8.encode(ro.device_pci)),
  )
  deployment_line(out, "readonly_device_name_hex", ro.device_name_hex)
  deployment_line(
    out,
    "readonly_device_memory",
    ro.device_memory_bytes.to_string(),
  )
  deployment_line(out, "kv_page_stride", kv.page_stride_bytes.to_string())
  @utf8.encode(out.to_string())
}

///|
pub fn export_digest_pinned_fused_production_runtime(
  artifacts : ApprovedFusedQkvReadOnlyProductionArtifacts,
) -> Bytes {
  fused_production_source_canonical(artifacts)
}

///|
fn deployment_parse(
  bytes : Bytes,
) -> Map[String, String] raise PagedCompositeArtifactError {
  if bytes.is_empty() || bytes.length() > 67108864 {
    raise InvalidPagedCompositeArtifact(Module)
  }
  let text = @utf8.decode(bytes) catch {
    _ => raise InvalidPagedCompositeArtifact(Module)
  }
  let fields : Map[String, String] = Map([])
  let lines = text.split("\n").to_array()
  if lines.length() < 3 || !lines[lines.length() - 1].is_empty() {
    raise InvalidPagedCompositeArtifact(Module)
  }
  for index = 0; index < lines.length() - 1; index = index + 1 {
    let line = lines[index]
    guard line.split_once("=") is Some((key, value)) &&
      !key.is_empty() &&
      !value.is_empty() &&
      !fields.contains(key.to_owned()) else {
      raise InvalidPagedCompositeArtifact(Module)
    }
    fields[key.to_owned()] = value.to_owned()
  }
  fields
}

///|
fn deployment_value(
  fields : Map[String, String],
  key : String,
) -> String raise PagedCompositeArtifactError {
  fields
  .get(key)
  .unwrap_or_else(() => raise InvalidPagedCompositeArtifact(Module))
}

///|
fn deployment_int(
  fields : Map[String, String],
  key : String,
) -> Int raise PagedCompositeArtifactError {
  let text = deployment_value(fields, key)
  let value : Int = @strconv.from_str(text) catch {
    _ => raise InvalidPagedCompositeArtifact(Module)
  }
  if value.to_string() != text {
    raise InvalidPagedCompositeArtifact(Module)
  }
  value
}

///|
fn deployment_i64(
  fields : Map[String, String],
  key : String,
) -> Int64 raise PagedCompositeArtifactError {
  let text = deployment_value(fields, key)
  let value : Int64 = @strconv.from_str(text) catch {
    _ => raise InvalidPagedCompositeArtifact(Module)
  }
  if value.to_string() != text {
    raise InvalidPagedCompositeArtifact(Module)
  }
  value
}

///|
fn deployment_hex(bytes : String) -> Bytes raise PagedCompositeArtifactError {
  if bytes.is_empty() || bytes.length() % 2 != 0 {
    raise InvalidPagedCompositeArtifact(Module)
  }
  fn nibble(unit : UInt16) -> Byte raise PagedCompositeArtifactError {
    if unit >= 48 && unit <= 57 {
      (unit.to_int() - 48).to_byte()
    } else if unit >= 97 && unit <= 102 {
      (unit.to_int() - 87).to_byte()
    } else {
      raise InvalidPagedCompositeArtifact(Module)
    }
  }
  Bytes::makei(bytes.length() / 2, index => {
    (nibble(bytes[index * 2]) << 4) | nibble(bytes[index * 2 + 1])
  })
}

///|
pub fn admit_digest_pinned_fused_production_runtime(
  canonical : Bytes,
  expected_sha256 : String,
) -> DigestPinnedFusedProductionRuntime raise PagedCompositeArtifactError {
  if !paged_composite_is_sha256(expected_sha256) ||
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)) != expected_sha256 {
    raise MismatchedPagedCompositeArtifact(Digest)
  }
  let f = deployment_parse(canonical)
  if deployment_value(f, "schema") != FUSED_PRODUCTION_SOURCE_SCHEMA {
    raise InvalidPagedCompositeArtifact(Module)
  }
  let fused_bytes = deployment_hex(deployment_value(f, "fused_source_hex"))
  let fused_digest = @crypto.bytes_to_hex_string(@crypto.sha256(fused_bytes))
  let (catalog, bundle) = @luna_fused_artifact_admission.admit_digest_pinned_fused_runtime_source(
    fused_bytes, fused_digest,
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(Module)
  }
  let digest_keys = [
    "readonly_manifest_sha256", "readonly_catalog_sha256", "readonly_candidate_sha256",
    "readonly_source_sha256", "readonly_recipe_sha256", "readonly_fallback_source_sha256",
    "readonly_fallback_recipe_sha256", "readonly_binding_sha256", "readonly_module_sha256",
    "readonly_physical_sha256", "readonly_policy_sha256", "readonly_toolchain_sha256",
    "readonly_driver_sha256", "readonly_runtime_device_sha256", "kv_model_content_sha256",
    "kv_model_plan_sha256", "kv_candidate_sha256", "kv_source_sha256", "kv_recipe_sha256",
    "kv_binding_sha256", "kv_module_sha256", "kv_physical_sha256", "kv_policy_sha256",
    "kv_fallback_source_sha256", "kv_fallback_recipe_sha256", "kv_toolchain_sha256",
    "kv_driver_sha256",
  ]
  for key in digest_keys {
    if !paged_composite_is_sha256(deployment_value(f, key)) {
      raise InvalidPagedCompositeArtifact(Digest)
    }
  }
  let module_bytes = deployment_hex(deployment_value(f, "readonly_module_hex"))
  if @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)) !=
    deployment_value(f, "readonly_module_sha256") {
    raise MismatchedPagedCompositeArtifact(Module)
  }
  let target = catalog.target()
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=deployment_int(f, "readonly_grid_x"),
    grid_y=deployment_int(f, "readonly_grid_y"),
    grid_z=deployment_int(f, "readonly_grid_z"),
    block_x=deployment_int(f, "readonly_block_x"),
    block_y=deployment_int(f, "readonly_block_y"),
    block_z=deployment_int(f, "readonly_block_z"),
    shared_memory_bytes=deployment_int(f, "readonly_shared_bytes"),
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(Module)
  }
  let symbol = @utf8.decode(
    deployment_hex(deployment_value(f, "readonly_symbol_hex")),
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(Module)
  }
  let device_uuid = @utf8.decode(
    deployment_hex(deployment_value(f, "readonly_device_uuid_hex")),
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(RuntimeDevice)
  }
  let device_pci = @utf8.decode(
    deployment_hex(deployment_value(f, "readonly_device_pci_hex")),
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(RuntimeDevice)
  }
  let ro : ApprovedPagedAttentionReadOnlyRuntimeIdentity = {
    manifest: deployment_value(f, "readonly_manifest_sha256"),
    catalog: deployment_value(f, "readonly_catalog_sha256"),
    candidate: deployment_value(f, "readonly_candidate_sha256"),
    source: deployment_value(f, "readonly_source_sha256"),
    recipe: deployment_value(f, "readonly_recipe_sha256"),
    fallback_source: deployment_value(f, "readonly_fallback_source_sha256"),
    fallback_recipe: deployment_value(f, "readonly_fallback_recipe_sha256"),
    compiled_binding: deployment_value(f, "readonly_binding_sha256"),
    module_digest: deployment_value(f, "readonly_module_sha256"),
    physical_evidence: deployment_value(f, "readonly_physical_sha256"),
    approved_policy: deployment_value(f, "readonly_policy_sha256"),
    toolchain: deployment_value(f, "readonly_toolchain_sha256"),
    driver: deployment_value(f, "readonly_driver_sha256"),
    target,
    operation: deployment_int(f, "readonly_operation"),
    activation: deployment_int(f, "readonly_activation"),
    input_row_width: deployment_int(f, "readonly_row_width"),
    profile_id: deployment_int(f, "readonly_profile_id"),
    profile_rows: deployment_int(f, "readonly_profile_rows"),
    profile_tokens: deployment_int(f, "readonly_profile_tokens"),
    profile_page_entries: deployment_int(f, "readonly_profile_pages"),
    tokens_per_page: deployment_int(f, "readonly_tokens_per_page"),
    page_stride_bytes: deployment_i64(f, "readonly_page_stride"),
    physical_page_capacity: deployment_i64(f, "readonly_page_capacity"),
    symbol,
    abi: RotatedQueryPagedKvReadOnlyProductionRawPointerV2,
    launch,
    module_bytes,
    dispatch_canary: deployment_int(f, "readonly_canary"),
    dispatch_canary_cells: deployment_int(f, "readonly_canary_cells"),
    device_uuid,
    device_pci,
    device_name_hex: deployment_value(f, "readonly_device_name_hex"),
    device_memory_bytes: deployment_i64(f, "readonly_device_memory"),
    runtime_device: deployment_value(f, "readonly_runtime_device_sha256"),
  }
  if ro.operation < 0 ||
    ro.activation < 0 ||
    ro.input_row_width <= 0 ||
    ro.dispatch_canary != 0 ||
    ro.dispatch_canary_cells != 0 ||
    ro.symbol.is_empty() {
    raise InvalidPagedCompositeArtifact(Module)
  }
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(
      deployment_int(f, "kv_profile_id"),
    ),
    max_query_rows=deployment_int(f, "kv_profile_rows"),
    max_query_tokens=deployment_int(f, "kv_profile_tokens"),
    max_page_table_entries=deployment_int(f, "kv_profile_pages"),
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(Fallback)
  }
  let kv = qualified_paged_kv_write_fallback_identity(
    model_content_sha256=deployment_value(f, "kv_model_content_sha256"),
    model_plan_sha256=deployment_value(f, "kv_model_plan_sha256"),
    candidate_sha256=deployment_value(f, "kv_candidate_sha256"),
    source_sha256=deployment_value(f, "kv_source_sha256"),
    recipe_sha256=deployment_value(f, "kv_recipe_sha256"),
    compiled_binding_sha256=deployment_value(f, "kv_binding_sha256"),
    module_sha256=deployment_value(f, "kv_module_sha256"),
    physical_evidence_sha256=deployment_value(f, "kv_physical_sha256"),
    approved_policy_sha256=deployment_value(f, "kv_policy_sha256"),
    fallback_source_sha256=deployment_value(f, "kv_fallback_source_sha256"),
    fallback_recipe_sha256=deployment_value(f, "kv_fallback_recipe_sha256"),
    operation=@plan.OperationId::new(deployment_int(f, "kv_operation")),
    activation=@plan.ActivationRef::new(deployment_int(f, "kv_activation")),
    row_width=deployment_int(f, "kv_row_width"),
    target~,
    profile~,
    tokens_per_page=deployment_int(f, "kv_tokens_per_page"),
    total_page_count=deployment_int(f, "kv_total_page_count"),
    page_stride_bytes=deployment_i64(f, "kv_page_stride"),
    toolchain_sha256=deployment_value(f, "kv_toolchain_sha256"),
    driver_sha256=deployment_value(f, "kv_driver_sha256"),
  ) catch {
    _ => raise InvalidPagedCompositeArtifact(Fallback)
  }
  let selected = select_fused_qkv_readonly_production_artifacts(
    catalog, bundle, ro, kv,
  ) catch {
    error => raise error
  }
  guard selected is Some(artifacts) else {
    raise MismatchedPagedCompositeArtifact(KernelAbi)
  }
  if fused_production_source_canonical(artifacts) != canonical {
    raise MismatchedPagedCompositeArtifact(Digest)
  }
  { artifacts, }
}

///|
pub fn DigestPinnedFusedProductionRuntime::fused_catalog(
  self : DigestPinnedFusedProductionRuntime,
) -> @luna_fused_artifact_admission.ApprovedFusedKernelCatalog {
  self.artifacts.fused_catalog
}

///|
pub fn DigestPinnedFusedProductionRuntime::fused_bundle(
  self : DigestPinnedFusedProductionRuntime,
) -> @luna_fused_artifact_admission.ApprovedFusedKernelBundle {
  self.artifacts.fused_bundle
}

///|
pub fn DigestPinnedFusedProductionRuntime::bind_live_device(
  self : DigestPinnedFusedProductionRuntime,
  live : @device.ContextDeviceIdentity,
) -> ApprovedFusedQkvReadOnlyProductionArtifacts raise PagedCompositeArtifactError {
  let ro = self.artifacts.readonly_identity
  if ro.device_uuid != live.uuid() ||
    ro.device_pci != live.pci() ||
    ro.device_name_hex != @crypto.bytes_to_hex_string(@utf8.encode(live.name())) ||
    ro.device_memory_bytes != live.total_memory_bytes() ||
    !ro.target.matches_capability(live.capability()) {
    raise MismatchedPagedCompositeArtifact(RuntimeDevice)
  }
  self.artifacts
}
