///|
fn v4_entry_point(
  claims : ManifestV4Claims,
  operation_index : Int,
) -> @catalog.AotKernelEntryPoint {
  let claim = claims.operations[operation_index]
  let module_claim = claims.modules[claim.module_index]
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=module_claim.digest,
      family_id=claim.family_id,
    ),
    entry_point_id=claim.entry_point_id,
  )
}

///|
fn derive_catalog_v4(
  claims : ManifestV4Claims,
  model_plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
) -> @catalog.ResolvedKernelCatalog raise ExecutionManifestFileError {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in model_plan.operations() {
    let claim = claims.operations[index]
    let family = v4_entry_point(claims, index).family()
    let entry = @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=operation_semantics(operation, layout),
      operation_execution_digest=model_plan.numeric_binding().operations()[index]
        .execution()
        .digest(),
      aot_entry_point=v4_entry_point(claims, index),
      compute_major=target.compute_major(),
      compute_minor=target.compute_minor(),
      implementation=ContentAddressedAot(family),
      workspace_bytes=claim.workspace_bytes,
      workspace_alignment=claim.workspace_alignment,
    ) catch {
      _ => raise AdmissionRejected(Catalog)
    }
    if !entries.contains(entry) {
      entries.push(entry)
    }
  }
  let catalog = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  ) catch {
    _ => raise AdmissionRejected(Catalog)
  }
  catalog.resolve_paged_v4(model_plan, layout, target) catch {
    _ => raise AdmissionRejected(Catalog)
  }
}

///|
fn derive_v4_operands(
  model_plan : @plan.ModelPlan,
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
        guard find_weight_region(static_operation, reference) is Some(region) &&
          static_plan.weight_layout().region(reference) == Some(region) &&
          model_plan.tensor(reference) is Some(tensor) &&
          tensor.role() == ParameterTensor else {
          raise AdmissionRejected(LaunchContracts)
        }
        result.push(
          make_operand(
            WeightInput(reference),
            region.byte_length(),
            static_plan.weight_alignment(),
          ),
        )
        match tensor.scale() {
          Some(scale) => {
            guard model_plan.tensor(scale.tensor()) is Some(metadata) &&
              metadata.role() == ScaleMetadataTensor &&
              static_plan.weight_layout().region(scale.tensor())
              is Some(scale_region) else {
              raise AdmissionRejected(LaunchContracts)
            }
            result.push(
              make_operand(
                WeightScaleInput(reference, scale),
                scale_region.byte_length(),
                static_plan.weight_alignment(),
              ),
            )
          }
          None => ()
        }
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
fn derive_contracts_v4(
  claims : ManifestV4Claims,
  model_plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  resolved : @catalog.ResolvedKernelCatalog,
  step_limits : @device_step.DeviceStepLimits,
  contract_limits : @launch_contract.LaunchContractLimits,
) -> @launch_contract.PagedV4LaunchContractSet raise ExecutionManifestFileError {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=claims.profile_id,
    max_query_rows=step_limits.max_rows(),
    max_query_tokens=step_limits.max_tokens(),
    max_page_table_entries=step_limits.max_pages(),
  ) catch {
    _ => raise AdmissionRejected(LaunchContracts)
  }
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for index, operation in model_plan.operations() {
    let claim = claims.operations[index]
    inputs.push(
      @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=claims.profile_id,
        operation_id=operation.id(),
        entry_point=v4_entry_point(claims, index),
        dimensions=claim.dimensions,
        operands=derive_v4_operands(
          model_plan,
          operation,
          static_plan.operations()[index],
          static_plan,
          memory_plan,
          layout,
          profile,
        ),
      ) catch {
        _ => raise AdmissionRejected(LaunchContracts)
      },
    )
  }
  @launch_contract.admit_paged_graph_v4(
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
fn derive_v4_artifact_source(
  claims : ManifestV4Claims,
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
