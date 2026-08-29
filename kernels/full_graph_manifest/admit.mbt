///|
fn manifest_module_index(
  modules : ArrayView[FullGraphManifestModule],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, item in modules {
    if item.digest == digest {
      return Some(index)
    }
  }
  None
}

///|
fn manifest_entry_index(
  entries : ArrayView[FullGraphManifestEntry],
  family_id : @catalog.AotKernelFamilyId,
  entry_point_id : @catalog.AotKernelEntryPointId,
) -> Int? {
  for index, entry in entries {
    if entry.family_id == family_id && entry.entry_point_id == entry_point_id {
      return Some(index)
    }
  }
  None
}

///|
fn insert_manifest_module(
  modules : Array[FullGraphManifestModule],
  item : FullGraphManifestModule,
) -> Unit {
  let mut insertion = modules.length()
  for index, existing in modules {
    if item.digest.sha256().compare(existing.digest.sha256()) < 0 {
      insertion = index
      break
    }
  }
  modules.insert(insertion, item)
}

///|
fn insert_manifest_entry(
  entries : Array[FullGraphManifestEntry],
  entry : FullGraphManifestEntry,
) -> Unit {
  let family = entry.family_id.as_int()
  let point = entry.entry_point_id.as_int()
  let mut insertion = entries.length()
  for index, existing in entries {
    let existing_family = existing.family_id.as_int()
    let existing_point = existing.entry_point_id.as_int()
    if family < existing_family ||
      (family == existing_family && point < existing_point) {
      insertion = index
      break
    }
  }
  entries.insert(insertion, entry)
}

///|
fn collect_manifest_modules(
  artifacts : ArrayView[FullGraphAotOperationArtifact],
  limits : FullGraphManifestLimits,
) -> Array[FullGraphManifestModule] raise FullGraphManifestError {
  let modules : Array[FullGraphManifestModule] = []
  let module_inputs : Array[@artifact_file.KernelModuleFileInput] = []
  let entry_inputs : Array[@artifact.KernelEntryPointInput] = []
  for index, artifact in artifacts {
    let entry_point = artifact.entry_point
    let family = entry_point.family()
    let digest = family.module_digest()
    if artifact.module_relative_path != "sha256/" + digest.sha256() + ".cubin" {
      raise Mismatch(rule=ModulePath, operation_index=index)
    }
    let entry = {
      family_id: family.family_id(),
      entry_point_id: entry_point.entry_point_id(),
      function_symbol: artifact.function_symbol,
    }
    match manifest_module_index(modules, digest) {
      Some(module_index) => {
        let item = modules[module_index]
        if item.path != artifact.module_relative_path {
          raise Mismatch(rule=ModulePath, operation_index=index)
        }
        match
          manifest_entry_index(
            item.entries,
            entry.family_id,
            entry.entry_point_id,
          ) {
          Some(entry_index) =>
            if item.entries[entry_index].function_symbol !=
              entry.function_symbol {
              raise Mismatch(rule=FunctionSymbol, operation_index=index)
            }
          None => insert_manifest_entry(item.entries, entry)
        }
      }
      None =>
        insert_manifest_module(modules, {
          digest,
          path: artifact.module_relative_path,
          entries: [entry],
        })
    }
  }
  for item in modules {
    module_inputs.push(
      @artifact_file.KernelModuleFileInput::new(
        digest=item.digest,
        path_identifier=item.path,
      ),
    )
    for entry in item.entries {
      entry_inputs.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=@catalog.AotKernelEntryPoint::new(
            family=@catalog.AotKernelFamily::new(
              module_digest=item.digest,
              family_id=entry.family_id,
            ),
            entry_point_id=entry.entry_point_id,
          ),
          function_symbol=entry.function_symbol,
        ),
      )
    }
  }
  @artifact.validate_entry_point_symbols(
    entry_inputs,
    limits.artifact_file.artifact(),
  ) catch {
    _ => raise Invalid(FunctionSymbol)
  }
  ignore(
    @artifact_file.KernelArtifactManifest::admit_files(
      module_inputs,
      entry_inputs,
      limits.artifact_file,
    ) catch {
      InvalidPath(..) => raise Invalid(ModulePath)
      _ => raise Invalid(Limits)
    },
  )
  modules
}

///|
fn validate_graph_contract(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : ArrayView[FullGraphAotOperationArtifact],
) -> Unit raise FullGraphManifestError {
  if contracts.scope() != FullGraph {
    raise Invalid(ContractScope)
  }
  if contracts.catalog_version() != @catalog.CatalogVersion::v3() {
    raise Invalid(CatalogVersion)
  }
  if contracts.model_identity() != model_plan.identity() {
    raise Invalid(ModelIdentity)
  }
  if contracts.device_target() != target {
    raise Invalid(DeviceTarget)
  }
  if contracts.profiles().length() != 1 {
    raise Invalid(Profile)
  }
  let operations = model_plan.operations()
  let admitted = contracts.contracts()
  if artifacts.length() != operations.length() ||
    admitted.length() != operations.length() {
    raise Invalid(OperationCoverage)
  }
  for index, operation in operations {
    let artifact = artifacts[index]
    let contract = admitted[index]
    if artifact.operation_id != operation.id() ||
      contract.operation_id() != operation.id() {
      raise Mismatch(rule=OperationOrder, operation_index=index)
    }
    if artifact.entry_point != contract.entry_point() {
      raise Mismatch(rule=EntryPoint, operation_index=index)
    }
    if artifact.dimensions != contract.dimensions() {
      raise Mismatch(rule=LaunchDimensions, operation_index=index)
    }
    if artifact.operands != contract.operands() {
      raise Mismatch(rule=OperandContract, operation_index=index)
    }
    if artifact.workspace_bytes != contract.workspace_bytes() ||
      artifact.workspace_alignment != contract.workspace_alignment() {
      raise Mismatch(rule=WorkspaceContract, operation_index=index)
    }
  }
}

///|
/// Produces canonical schema-v2 bytes accepted by the existing authenticated
/// execution-manifest loader.
///
/// The producer does not derive launch geometry or operand sizes. It requires
/// exact equality between offline artifact metadata and a separately admitted
/// FullGraph launch-contract set before serialization.
pub fn produce_execution_manifest_v2(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : ArrayView[FullGraphAotOperationArtifact],
  limits : FullGraphManifestLimits,
) -> FullGraphExecutionManifestInput raise FullGraphManifestError {
  validate_graph_contract(model_plan, target, contracts, artifacts)
  let modules = collect_manifest_modules(artifacts, limits)
  serialize_full_graph_manifest(
    model_plan, target, contracts, artifacts, modules, limits,
  )
}
