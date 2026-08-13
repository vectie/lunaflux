///|
fn parse_entry_claim(
  value : Json,
  module_index : Int,
  entry_index : Int,
  maximum_symbol_bytes : Int,
) -> ManifestEntryClaim raise ExecutionManifestFileError {
  guard value is Object(object) else {
    raise invalid_field(Module, ExpectedObject, index=module_index)
  }
  validate_fields(object, Module, ["family_id", "entry_point_id", "symbol"])
  let family_id = @catalog.AotKernelFamilyId::from_stable_int(
    require_nonnegative_int(object, "family_id", Module, index=entry_index),
  ) catch {
    _ => raise invalid_field(Module, UnsupportedValue, index=entry_index)
  }
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(
    require_nonnegative_int(object, "entry_point_id", Module, index=entry_index),
  ) catch {
    _ => raise invalid_field(Module, UnsupportedValue, index=entry_index)
  }
  let symbol = require_string(object, "symbol", Module, index=entry_index)
  if symbol.length() > maximum_symbol_bytes {
    raise invalid_field(Module, CountExceeded, index=entry_index)
  }
  { family_id, entry_point_id, symbol }
}

///|
fn parse_module_claims(
  root : Map[String, Json],
  limits : ExecutionManifestFileLimits,
) -> Array[ManifestModuleClaim] raise ExecutionManifestFileError {
  let values = require_array(root, "modules", Root)
  if values.is_empty() || values.length() > limits.max_modules() {
    raise invalid_field(Root, CountExceeded)
  }
  let result : Array[ManifestModuleClaim] = []
  let paths : Map[String, Bool] = Map([])
  let mut entry_count = 0
  let artifact_limits = limits.artifact_file().artifact()
  for module_index, value in values {
    guard value is Object(object) else {
      raise invalid_field(Module, ExpectedObject, index=module_index)
    }
    validate_fields(object, Module, ["sha256", "path", "entry_points"])
    let digest_text = require_string(
      object,
      "sha256",
      Module,
      index=module_index,
    )
    let digest = @catalog.AotArtifactDigest::from_sha256(digest_text) catch {
      _ => raise invalid_field(Module, InvalidSha256, index=module_index)
    }
    if module_index > 0 &&
      !lexically_before(result[module_index - 1].digest.sha256(), digest_text) {
      raise invalid_field(Module, InvalidOrder, index=module_index)
    }
    let path = require_string(object, "path", Module, index=module_index)
    if paths.contains(path) {
      raise invalid_field(Module, DuplicateValue, index=module_index)
    }
    paths[path] = true
    let entry_values = require_array(
      object,
      "entry_points",
      Module,
      index=module_index,
    )
    if entry_values.is_empty() ||
      entry_values.length() > artifact_limits.max_entry_points() - entry_count {
      raise invalid_field(Module, CountExceeded, index=module_index)
    }
    let entries : Array[ManifestEntryClaim] = []
    for entry_index, entry_value in entry_values {
      let entry = parse_entry_claim(
        entry_value,
        module_index,
        entry_index,
        artifact_limits.max_function_symbol_bytes(),
      )
      if entry_index > 0 {
        let previous = entries[entry_index - 1]
        let family = entry.family_id.as_int()
        let previous_family = previous.family_id.as_int()
        if family < previous_family ||
          (
            family == previous_family &&
            entry.entry_point_id.as_int() <= previous.entry_point_id.as_int()
          ) {
          raise invalid_field(Module, InvalidOrder, index=entry_index)
        }
      }
      entries.push(entry)
      entry_count += 1
    }
    result.push({ digest, path, entries })
  }
  result
}

///|
fn parse_dimensions(
  operation : Map[String, Json],
  operation_index : Int,
) -> @launch_contract.AotLaunchDimensions raise ExecutionManifestFileError {
  let value = require_object(
    operation,
    "dimensions",
    Operation,
    index=operation_index,
  )
  validate_fields(value, Dimensions, [
    "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_memory_bytes",
  ])
  @launch_contract.AotLaunchDimensions::new(
    grid_x=require_nonnegative_int(value, "grid_x", Dimensions),
    grid_y=require_nonnegative_int(value, "grid_y", Dimensions),
    grid_z=require_nonnegative_int(value, "grid_z", Dimensions),
    block_x=require_nonnegative_int(value, "block_x", Dimensions),
    block_y=require_nonnegative_int(value, "block_y", Dimensions),
    block_z=require_nonnegative_int(value, "block_z", Dimensions),
    shared_memory_bytes=require_nonnegative_int(
      value,
      "shared_memory_bytes",
      Dimensions,
    ),
  ) catch {
    _ =>
      raise invalid_field(Dimensions, UnsupportedValue, index=operation_index)
  }
}

///|
fn module_entry_index(
  module_claim : ManifestModuleClaim,
  family_id : @catalog.AotKernelFamilyId,
  entry_point_id : @catalog.AotKernelEntryPointId,
) -> Int? {
  let family = family_id.as_int()
  let entry_point = entry_point_id.as_int()
  let mut lower = 0
  let mut upper = module_claim.entries.length()
  while lower < upper {
    let middle = lower + (upper - lower) / 2
    let candidate = module_claim.entries[middle]
    let candidate_family = candidate.family_id.as_int()
    let candidate_entry = candidate.entry_point_id.as_int()
    if candidate_family < family ||
      (candidate_family == family && candidate_entry < entry_point) {
      lower = middle + 1
    } else {
      upper = middle
    }
  }
  if lower < module_claim.entries.length() {
    let candidate = module_claim.entries[lower]
    if candidate.family_id == family_id &&
      candidate.entry_point_id == entry_point_id {
      return Some(lower)
    }
  }
  None
}

///|
fn parse_operation_claims(
  root : Map[String, Json],
  modules : ArrayView[ManifestModuleClaim],
  model_plan : @plan.ModelPlan,
  limits : ExecutionManifestFileLimits,
) -> Array[ManifestOperationClaim] raise ExecutionManifestFileError {
  let values = require_array(root, "operations", Root)
  let expected = model_plan.operations()
  if values.length() != expected.length() ||
    values.length() > limits.max_operations() {
    raise invalid_field(Root, CountExceeded, index=1)
  }
  let result : Array[ManifestOperationClaim] = []
  let referenced : Array[Array[Bool]] = []
  for module_claim in modules {
    referenced.push(Array::make(module_claim.entries.length(), false))
  }
  for index, value in values {
    guard value is Object(object) else {
      raise invalid_field(Operation, ExpectedObject, index~)
    }
    validate_fields(object, Operation, [
      "operation_id", "module_index", "family_id", "entry_point_id", "workspace_bytes",
      "workspace_alignment", "dimensions",
    ])
    let operation_id = require_nonnegative_int(
      object,
      "operation_id",
      Operation,
      index~,
    )
    if operation_id != expected[index].id().as_int() {
      raise invalid_field(Operation, InvalidOrder, index~)
    }
    let module_index = require_nonnegative_int(
      object,
      "module_index",
      Operation,
      index~,
    )
    if module_index >= modules.length() {
      raise invalid_field(Operation, UnsupportedValue, index~)
    }
    let family_id = @catalog.AotKernelFamilyId::from_stable_int(
      require_nonnegative_int(object, "family_id", Operation, index~),
    ) catch {
      _ => raise invalid_field(Operation, UnsupportedValue, index~)
    }
    let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(
      require_nonnegative_int(object, "entry_point_id", Operation, index~),
    ) catch {
      _ => raise invalid_field(Operation, UnsupportedValue, index~)
    }
    guard module_entry_index(modules[module_index], family_id, entry_point_id)
      is Some(module_entry_index) else {
      raise invalid_field(Operation, UnsupportedValue, index~)
    }
    referenced[module_index][module_entry_index] = true
    let workspace_bytes = require_nonnegative_int64(
      object,
      "workspace_bytes",
      Operation,
      index~,
    )
    let workspace_alignment = require_nonnegative_int64(
      object,
      "workspace_alignment",
      Operation,
      index~,
    )
    result.push({
      module_index,
      family_id,
      entry_point_id,
      workspace_bytes,
      workspace_alignment,
      dimensions: parse_dimensions(object, index),
    })
  }
  for module_index, module_claim in modules {
    let mut module_referenced = false
    for entry_index, _ in module_claim.entries {
      if !referenced[module_index][entry_index] {
        raise invalid_field(Module, UnsupportedValue, index=module_index)
      }
      module_referenced = true
    }
    if !module_referenced {
      raise invalid_field(Module, UnsupportedValue, index=module_index)
    }
  }
  result
}
