///|
fn validate_bootstrap_inputs(
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  device_ordinal : Int,
  limits : DeviceBootstrapLimits,
) -> Unit raise DeviceBootstrapError {
  if device_ordinal < 0 || device_ordinal > 1023 {
    raise InvalidBootstrap(DeviceOrdinal)
  }
  if blueprint.identity() != artifacts.model_identity() {
    raise InvalidBootstrap(Identity)
  }
  if blueprint.target() != artifacts.device_target() {
    raise InvalidBootstrap(Target)
  }
  if blueprint.catalog_version() != @catalog.CatalogVersion::v3() ||
    artifacts.catalog_version() != blueprint.catalog_version() {
    raise InvalidBootstrap(CatalogVersion)
  }
  if blueprint.scope() != FullGraph {
    raise InvalidBootstrap(Scope)
  }
  if artifacts.module_count() <= 0 ||
    artifacts.module_count() > limits.max_modules ||
    artifacts.entry_point_count() <= 0 ||
    artifacts.entry_point_count() > limits.max_entry_points ||
    blueprint.steps().is_empty() ||
    blueprint.steps().length() > limits.max_steps {
    raise InvalidBootstrap(Capacity)
  }
  for step in blueprint.steps() {
    if step.operands().is_empty() ||
      step.operands().length() > limits.max_operands_per_step ||
      step.artifact_module_index() < 0 ||
      step.artifact_module_index() >= artifacts.module_count() {
      raise InvalidBootstrap(Blueprint)
    }
    guard artifacts.entry_point(step.entry_point()) is Some(admitted) else {
      raise InvalidBootstrap(Artifact)
    }
    if admitted.module_index() != step.artifact_module_index() ||
      admitted.function_symbol() != step.function_symbol() ||
      @utf8.encode(step.function_symbol()).length() > limits.max_symbol_bytes {
      raise InvalidBootstrap(Artifact)
    }
  }
}

///|
fn encode_bootstrap(
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  device_ordinal : Int,
  limits : DeviceBootstrapLimits,
) -> Bytes raise DeviceBootstrapError {
  let encoder : BootstrapEncoder = {
    bytes: [],
    maximum: limits.max_canonical_bytes,
  }
  encoder.append_string("lunaflux.device-worker-bootstrap.v2")
  encoder.append_string(blueprint.identity().content().as_hex())
  encoder.append_string(blueprint.identity().plan().as_hex())
  encoder.append_int(device_ordinal)
  let target = blueprint.target()
  encoder.append_int(target.compute_major())
  encoder.append_int(target.compute_minor())
  encoder.append_int(if target.supports_bf16() { 1 } else { 0 })
  encoder.append_int(if target.supports_cublas_lt() { 1 } else { 0 })
  encoder.append_int(blueprint.catalog_version().as_int())
  let profile = blueprint.profile()
  encoder.append_int(profile.id().as_int())
  encoder.append_int(profile.max_query_rows())
  encoder.append_int(profile.max_query_tokens())
  encoder.append_int(profile.max_page_table_entries())
  let device_limits = blueprint.limits()
  encoder.append_int(device_limits.max_rows())
  encoder.append_int(device_limits.max_tokens())
  encoder.append_int(device_limits.max_pages())
  encoder.append_int(device_limits.max_sequence_tokens())
  encoder.append_int(device_limits.max_token_id())
  let layout = blueprint.layout()
  let geometry = layout.geometry()
  encoder.append_int(layout.stable_layout_version())
  encoder.append_int(geometry.layer_count())
  encoder.append_int(geometry.kv_head_count())
  encoder.append_int(geometry.head_dimension())
  encoder.append_int(layout.tokens_per_page())
  encoder.append_int(layout.total_page_count())
  encoder.append_i64(layout.segment_alignment_bytes())
  encoder.append_i64(layout.page_logical_bytes())
  encoder.append_i64(layout.page_stride_bytes())
  encoder.append_i64(layout.component_stride_bytes())
  encoder.append_i64(layout.layer_stride_bytes())
  encoder.append_i64(layout.arena_bytes())
  encoder.append_i64(blueprint.weight_arena_bytes())
  encoder.append_i64(blueprint.weight_alignment())
  encoder.append_i64(blueprint.activation_workspace_arena_bytes())

  encoder.append_int(artifacts.module_count())
  for artifact_module in artifacts.modules() {
    encoder.append_string(artifact_module.digest().sha256())
    encoder.append_int(artifact_module.module_bytes().length())
  }
  encoder.append_int(artifacts.entry_point_count())
  for entry in artifacts.entry_points() {
    encoder.append_entry_identity(entry.identity())
    encoder.append_int(entry.module_index())
    encoder.append_string(entry.function_symbol())
  }
  encoder.append_int(blueprint.steps().length())
  for step in blueprint.steps() {
    encoder.append_int(step.operation_id().as_int())
    encoder.append_int(operation_kind_code(step.operation_kind()))
    match step.kv_layer() {
      Some(layer) => {
        encoder.append_int(1)
        encoder.append_int(layer.as_int())
      }
      None => encoder.append_int(0)
    }
    encoder.append_int(step.capability().as_int())
    encoder.append_int(step.semantic_version().as_int())
    encoder.append_entry_identity(step.entry_point())
    encoder.append_int(step.artifact_module_index())
    encoder.append_string(step.function_symbol())
    encoder.append_dimensions(step.dimensions())
    encoder.append_int(step.operands().length())
    for operand in step.operands() {
      encoder.append_role(operand.role())
      encoder.append_space(operand.space())
      encoder.append_i64(operand.offset_bytes())
      encoder.append_i64(operand.byte_count())
      encoder.append_i64(operand.alignment())
    }
  }
  Bytes::from_array(encoder.bytes)
}

///|
/// Derives one content identity for the complete admitted device-worker
/// bootstrap. Module bytes are represented by their already-verified SHA-256;
/// every stable symbol, launch, layout, and physical operand binding is encoded
/// directly. The result owns no module bytes or device resource.
pub fn admit_device_worker_bootstrap(
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  device_ordinal~ : Int,
  limits : DeviceBootstrapLimits,
) -> DeviceWorkerBootstrapManifest raise DeviceBootstrapError {
  validate_bootstrap_inputs(blueprint, artifacts, device_ordinal, limits)
  let canonical = encode_bootstrap(blueprint, artifacts, device_ordinal, limits)
  let digest = @worker_wire.WorkerBootstrapDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  ) catch {
    _ => raise InvalidBootstrap(CanonicalBytes)
  }
  {
    digest,
    identity: blueprint.identity(),
    device_ordinal,
    target: blueprint.target(),
    catalog_version: blueprint.catalog_version(),
    limits: blueprint.limits(),
    module_count: artifacts.module_count(),
    entry_point_count: artifacts.entry_point_count(),
    step_count: blueprint.steps().length(),
  }
}
