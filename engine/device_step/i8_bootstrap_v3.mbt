///|
fn BootstrapEncoder::append_i8_role(
  self : BootstrapEncoder,
  role : @launch_contract.AotOperandRole,
) -> Unit raise DeviceBootstrapError {
  match role {
    WeightScaleInput(weight, scale) => {
      self.append_int(8)
      self.append_int(weight.as_int())
      self.append_int(scale.tensor().as_int())
    }
    _ => self.append_role(role)
  }
}

///|
fn validate_i8_bootstrap_v3(
  blueprint : I8PagedExecutionBlueprint,
  device_ordinal : Int,
  limits : DeviceBootstrapLimits,
) -> Unit raise DeviceBootstrapError {
  let artifacts = blueprint.artifacts
  if device_ordinal < 0 || device_ordinal > 1023 {
    raise InvalidBootstrap(DeviceOrdinal)
  }
  if artifacts.model_identity() != blueprint.identity() {
    raise InvalidBootstrap(Identity)
  }
  if artifacts.device_target() != blueprint.target() {
    raise InvalidBootstrap(Target)
  }
  if blueprint.catalog_version() != @catalog.CatalogVersion::v4() ||
    artifacts.catalog_version() != blueprint.catalog_version() {
    raise InvalidBootstrap(CatalogVersion)
  }
  if artifacts.module_count() <= 0 ||
    artifacts.module_count() > limits.max_modules ||
    artifacts.entry_point_count() <= 0 ||
    artifacts.entry_point_count() > limits.max_entry_points ||
    blueprint.steps().is_empty() ||
    blueprint.steps().length() > limits.max_steps {
    raise InvalidBootstrap(Capacity)
  }
  let mut scale_roles = 0
  for step in blueprint.steps() {
    if step.semantic_version() != @catalog.KernelSemanticVersion::v4() ||
      step.operands().is_empty() ||
      step.operands().length() > limits.max_operands_per_step ||
      step.artifact_module_index() < 0 ||
      step.artifact_module_index() >= artifacts.module_count() {
      raise InvalidBootstrap(Blueprint)
    }
    guard artifacts.entry_point(step.entry_point()) is Some(entry) else {
      raise InvalidBootstrap(Artifact)
    }
    if entry.module_index() != step.artifact_module_index() ||
      entry.function_symbol() != step.function_symbol() ||
      @utf8.encode(step.function_symbol()).length() > limits.max_symbol_bytes {
      raise InvalidBootstrap(Artifact)
    }
    for operand in step.operands() {
      if operand.role() is WeightScaleInput(_, _) {
        scale_roles += 1
        if operand.space() != WeightArena {
          raise InvalidBootstrap(Blueprint)
        }
      }
    }
  }
  if scale_roles <= 0 {
    raise InvalidBootstrap(Blueprint)
  }
}

///|
fn encode_i8_bootstrap_v3(
  blueprint : I8PagedExecutionBlueprint,
  device_ordinal : Int,
  limits : DeviceBootstrapLimits,
) -> Bytes raise DeviceBootstrapError {
  let artifacts = blueprint.artifacts
  let encoder : BootstrapEncoder = {
    bytes: [],
    maximum: limits.max_canonical_bytes,
  }
  encoder.append_string("lunaflux.device-worker-bootstrap.v3\u{0000}")
  encoder.append_int(3)
  encoder.append_string(blueprint.identity().content().as_hex())
  encoder.append_string(blueprint.identity().plan().as_hex())
  encoder.append_string(blueprint.numeric_schema_digest().as_hex())
  encoder.append_string(blueprint.i8_admission_digest().as_hex())
  encoder.append_string(blueprint.device_numeric_digest().as_hex())
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
  let kv = blueprint.layout()
  let geometry = kv.geometry()
  encoder.append_int(kv.stable_layout_version())
  encoder.append_int(geometry.layer_count())
  encoder.append_int(geometry.kv_head_count())
  encoder.append_int(geometry.head_dimension())
  encoder.append_int(kv.tokens_per_page())
  encoder.append_int(kv.total_page_count())
  encoder.append_i64(kv.segment_alignment_bytes())
  encoder.append_i64(kv.page_logical_bytes())
  encoder.append_i64(kv.page_stride_bytes())
  encoder.append_i64(kv.component_stride_bytes())
  encoder.append_i64(kv.layer_stride_bytes())
  encoder.append_i64(kv.arena_bytes())
  let weights = blueprint.weight_layout()
  encoder.append_string(
    match weights.numeric_schema_digest() {
      Some(digest) => digest.as_hex()
      None => raise InvalidBootstrap(Blueprint)
    },
  )
  encoder.append_i64(weights.alignment_bytes())
  encoder.append_i64(weights.arena_bytes())
  encoder.append_i64(weights.materialized_bytes())
  encoder.append_int(weights.length())
  for index in 0..<weights.length() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise InvalidBootstrap(Blueprint)
    }
    guard weights.region(reference) is Some(region) else {
      raise InvalidBootstrap(Blueprint)
    }
    encoder.append_int(region.reference().as_int())
    encoder.append_i64(region.offset_bytes())
    encoder.append_i64(region.byte_length())
  }
  encoder.append_i64(blueprint.activation_workspace_arena_bytes())
  encoder.append_int(artifacts.module_count())
  for module_image in artifacts.modules() {
    encoder.append_string(module_image.digest().sha256())
    encoder.append_int(module_image.module_bytes().length())
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
    encoder.append_string(step.operation_execution_digest().as_hex())
    encoder.append_entry_identity(step.entry_point())
    encoder.append_int(step.artifact_module_index())
    encoder.append_string(step.function_symbol())
    encoder.append_dimensions(step.dimensions())
    encoder.append_int(step.operands().length())
    for operand in step.operands() {
      encoder.append_i8_role(operand.role())
      encoder.append_space(operand.space())
      encoder.append_i64(operand.offset_bytes())
      encoder.append_i64(operand.byte_count())
      encoder.append_i64(operand.alignment())
    }
  }
  Bytes::from_array(encoder.bytes)
}

///|
/// Derives bootstrap-v3 identity from every production-I8 numeric, layout,
/// catalog-v4, launch, and admitted artifact fact. Its parallel startup method
/// binds only worker protocol identity and envelopes; exact loaded-weight
/// authority must still join before executor preparation.
pub fn admit_i8_device_worker_bootstrap_v3(
  blueprint : I8PagedExecutionBlueprint,
  device_ordinal~ : Int,
  limits : DeviceBootstrapLimits,
) -> I8DeviceWorkerBootstrapManifestV3 raise DeviceBootstrapError {
  validate_i8_bootstrap_v3(blueprint, device_ordinal, limits)
  let canonical = encode_i8_bootstrap_v3(blueprint, device_ordinal, limits)
  let digest = @worker_wire.WorkerBootstrapDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  ) catch {
    _ => raise InvalidBootstrap(CanonicalBytes)
  }
  {
    digest,
    identity: blueprint.identity(),
    numeric_schema_digest: blueprint.numeric_schema_digest(),
    i8_admission_digest: blueprint.i8_admission_digest(),
    device_numeric_digest: blueprint.device_numeric_digest(),
    device_ordinal,
    target: blueprint.target(),
    catalog_version: blueprint.catalog_version(),
    limits: blueprint.limits(),
    module_count: blueprint.artifacts.module_count(),
    entry_point_count: blueprint.artifacts.entry_point_count(),
    step_count: blueprint.steps().length(),
  }
}
