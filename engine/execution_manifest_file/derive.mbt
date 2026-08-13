///|
fn operation_semantics(
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
) -> @catalog.KernelExecutionSemantics raise ExecutionManifestFileError {
  let kv_layout = match operation.persistent_state() {
    NoPersistentState => None
    KvCacheReadWrite(_) =>
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=layout.geometry().layout(),
          stable_version=layout.stable_layout_version(),
          tokens_per_page=layout.tokens_per_page(),
        ) catch {
          _ => raise AdmissionRejected(Catalog)
        },
      )
  }
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  ) catch {
    _ => raise AdmissionRejected(Catalog)
  }
}

///|
fn derive_catalog(
  claims : ManifestClaims,
  model_plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
) -> @catalog.ResolvedKernelCatalog raise ExecutionManifestFileError {
  let entries : Array[@catalog.CatalogEntry] = []
  let by_family : Map[@catalog.AotKernelFamily, @catalog.CatalogEntry] = Map([])
  for index, operation in model_plan.operations() {
    let claim = claims.operations[index]
    let module_claim = claims.modules[claim.module_index]
    let entry = @catalog.CatalogEntry::new_paged_v3(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=operation_semantics(operation, layout),
      compute_major=target.compute_major(),
      compute_minor=target.compute_minor(),
      implementation=ContentAddressedAot(
        @catalog.AotKernelFamily::new(
          module_digest=module_claim.digest,
          family_id=claim.family_id,
        ),
      ),
      workspace_bytes=claim.workspace_bytes,
      workspace_alignment=claim.workspace_alignment,
    ) catch {
      _ => raise AdmissionRejected(Catalog)
    }
    guard entry.implementation() is ContentAddressedAot(family) else {
      raise AdmissionRejected(Catalog)
    }
    match by_family.get(family) {
      Some(existing) =>
        if existing != entry {
          raise AdmissionRejected(Catalog)
        }
      None => {
        by_family[family] = entry
        entries.push(entry)
      }
    }
  }
  let catalog = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v3(),
    entries~,
  ) catch {
    _ => raise AdmissionRejected(Catalog)
  }
  catalog.resolve_paged(model_plan, layout, target) catch {
    _ => raise AdmissionRejected(Catalog)
  }
}

///|
fn runtime_bytes(
  input : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match input {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn activation_bytes(
  memory : @device_memory.DeviceMemoryPlan,
  reference : @plan.ActivationRef,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 raise ExecutionManifestFileError {
  guard memory.region(reference) is Some(region) else {
    raise AdmissionRejected(LaunchContracts)
  }
  let rows = memory.token_rows()
  if rows <= 0L || region.logical_bytes() % rows != 0L {
    raise AdmissionRejected(LaunchContracts)
  }
  region.logical_bytes() / rows * profile.max_query_tokens().to_int64()
}

///|
fn make_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise ExecutionManifestFileError {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~) catch {
    _ => raise AdmissionRejected(LaunchContracts)
  }
}

///|
fn find_weight_region(
  operation : @device_plan.StaticOperation,
  reference : @plan.TensorRef,
) -> @device_materialize.DeviceWeightRegion? {
  for input in operation.inputs() {
    if input is WeightRegion(region) && region.reference() == reference {
      return Some(region)
    }
  }
  None
}

///|
fn derive_operands(
  operation : @plan.PlanOperation,
  static_operation : @device_plan.StaticOperation,
  static_plan : @device_plan.StaticDevicePlan,
  memory : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
) -> Array[@launch_contract.AotOperand] raise ExecutionManifestFileError {
  let result : Array[@launch_contract.AotOperand] = []
  for runtime in operation.runtime_inputs() {
    result.push(
      make_operand(RuntimeInput(runtime), runtime_bytes(runtime, profile), 4L),
    )
  }
  for input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        result.push(
          make_operand(
            ExternalInput(reference),
            profile.max_query_tokens().to_int64() * 4L,
            4L,
          ),
        )
      Tensor(reference) => {
        guard find_weight_region(static_operation, reference) is Some(region) else {
          raise AdmissionRejected(LaunchContracts)
        }
        result.push(
          make_operand(
            WeightInput(reference),
            region.byte_length(),
            static_plan.weight_alignment(),
          ),
        )
      }
      Activation(reference) =>
        result.push(
          make_operand(
            ActivationInput(reference),
            activation_bytes(memory, reference, profile),
            memory.limits().activation_alignment(),
          ),
        )
    }
  }
  for output in operation.outputs() {
    result.push(
      make_operand(
        ActivationOutput(output),
        activation_bytes(memory, output, profile),
        memory.limits().activation_alignment(),
      ),
    )
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      result.push(
        make_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
      result.push(
        make_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
    }
  }
  if static_operation.binding().workspace_bytes() > 0L {
    result.push(
      make_operand(
        Workspace,
        static_operation.binding().workspace_bytes(),
        static_operation.binding().workspace_alignment(),
      ),
    )
  }
  result
}

///|
fn derive_contracts(
  claims : ManifestClaims,
  model_plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  resolved : @catalog.ResolvedKernelCatalog,
  step_limits : @device_step.DeviceStepLimits,
  contract_limits : @launch_contract.LaunchContractLimits,
) -> @launch_contract.PagedKvLaunchContractSet raise ExecutionManifestFileError {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=claims.profile_id,
    max_query_rows=step_limits.max_rows(),
    max_query_tokens=step_limits.max_tokens(),
    max_page_table_entries=step_limits.max_pages(),
  ) catch {
    _ => raise AdmissionRejected(LaunchContracts)
  }
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for index, operation in model_plan.operations() {
    let claim = claims.operations[index]
    let module_claim = claims.modules[claim.module_index]
    let entry_point = @catalog.AotKernelEntryPoint::new(
      family=@catalog.AotKernelFamily::new(
        module_digest=module_claim.digest,
        family_id=claim.family_id,
      ),
      entry_point_id=claim.entry_point_id,
    )
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=claims.profile_id,
        operation_id=operation.id(),
        entry_point~,
        dimensions=claim.dimensions,
        operands=derive_operands(
          operation,
          static_plan.operations()[index],
          static_plan,
          memory_plan,
          layout,
          profile,
        ),
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    model_plan,
    resolved,
    layout,
    [profile],
    inputs,
    contract_limits,
  ) catch {
    _ => raise AdmissionRejected(LaunchContracts)
  }
}

///|
fn derive_artifact_source(
  claims : ManifestClaims,
  limits : @artifact_file.KernelArtifactFileLimits,
) -> @artifact_file.KernelArtifactManifest raise ExecutionManifestFileError {
  let modules : Array[@artifact_file.KernelModuleFileInput] = []
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for module_claim in claims.modules {
    modules.push(
      @artifact_file.KernelModuleFileInput::new(
        digest=module_claim.digest,
        path_identifier=module_claim.path,
      ),
    )
    for entry in module_claim.entries {
      entries.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=@catalog.AotKernelEntryPoint::new(
            family=@catalog.AotKernelFamily::new(
              module_digest=module_claim.digest,
              family_id=entry.family_id,
            ),
            entry_point_id=entry.entry_point_id,
          ),
          function_symbol=entry.symbol,
        ),
      )
    }
  }
  @artifact.validate_entry_point_symbols(entries, limits.artifact()) catch {
    _ => raise AdmissionRejected(ArtifactSource)
  }
  @artifact_file.KernelArtifactManifest::admit_files(modules, entries, limits) catch {
    _ => raise AdmissionRejected(ArtifactSource)
  }
}
