///|
priv struct I8BlueprintFixture {
  plan : @plan.ModelPlan
  device_numeric : @device.DeviceNumericCapability
  policy : @i8_admission.I8ProductionPolicyAdmission
  layout : @device_layout.DeviceKvLayout
  static_plan : @device_plan.StaticDevicePlan
  memory_plan : @device_memory.DeviceMemoryPlan
  profile : @launch_contract.PagedKernelProfile
  contracts : @launch_contract.PagedV4LaunchContractSet
  artifacts : @artifact.KernelArtifactBundle
  blueprint : I8PagedExecutionBlueprint
}

///|
fn i8_fixture_device() -> @device.DeviceNumericCapability raise {
  @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    @device.DeviceCapability::from_observed_facts(
      compute_major=8,
      compute_minor=9,
      total_memory_bytes=8L * 1024L * 1024L * 1024L,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn i8_fixture_resolved(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
) -> @catalog.ResolvedKernelCatalog raise {
  let module_digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    let mut representative = index
    for earlier in 0..<index {
      let candidate = plan.operations()[earlier]
      if candidate.kind() == operation.kind() &&
        candidate.shape() == operation.shape() &&
        candidate.required_capability() == operation.required_capability() &&
        candidate.kv_layer() == operation.kv_layer() &&
        candidate.runtime_inputs() == operation.runtime_inputs() &&
        candidate.persistent_state() == operation.persistent_state() &&
        candidate.workspace_bytes() == operation.workspace_bytes() &&
        plan.numeric_binding().operations()[earlier].execution().digest() ==
        plan.numeric_binding().operations()[index].execution().digest() {
        representative = earlier
        break
      }
    }
    let family = @catalog.AotKernelFamily::new(
      module_digest~,
      family_id=@catalog.AotKernelFamilyId::from_stable_int(representative + 1),
    )
    let kv_layout = if operation.persistent_state() is KvCacheReadWrite(_) {
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=layout.geometry().layout(),
          stable_version=layout.stable_layout_version(),
          tokens_per_page=layout.tokens_per_page(),
        ),
      )
    } else {
      None
    }
    let entry = @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=@catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=operation.kv_layer(),
        runtime_inputs=operation.runtime_inputs(),
        persistent_state=operation.persistent_state(),
        kv_layout~,
      ),
      operation_execution_digest=plan.numeric_binding().operations()[index]
        .execution()
        .digest(),
      aot_entry_point=@catalog.AotKernelEntryPoint::new(
        family~,
        entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
          representative + 1,
        ),
      ),
      compute_major=target.compute_major(),
      compute_minor=target.compute_minor(),
      implementation=ContentAddressedAot(family),
      workspace_bytes=operation.workspace_bytes(),
      workspace_alignment=if operation.workspace_bytes() > 0L {
        16L
      } else {
        1L
      },
    )
    if !entries.contains(entry) {
      entries.push(entry)
    }
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v4(), entries~).resolve_paged_v4(
    plan, layout, target,
  )
}

///|
fn i8_fixture_memory(
  static_plan : @device_plan.StaticDevicePlan,
  maximum? : Int64 = 65536L,
) -> @device_memory.DeviceMemoryPlan raise {
  @device_memory.plan(
    static_plan,
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=maximum,
    ),
  )
}

///|
fn i8_fixture_operands(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  static_operation : @device_plan.StaticOperation,
  static_plan : @device_plan.StaticDevicePlan,
  memory : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
) -> Array[@launch_contract.AotOperand] raise {
  let result : Array[@launch_contract.AotOperand] = []
  for runtime in operation.runtime_inputs() {
    result.push(
      blueprint_operand(
        RuntimeInput(runtime),
        full_runtime_bytes(runtime, profile),
        4L,
      ),
    )
  }
  for input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        result.push(
          blueprint_operand(
            ExternalInput(reference),
            profile.max_query_tokens().to_int64() * 4L,
            4L,
          ),
        )
      Tensor(reference) => {
        guard static_plan.weight_layout().region(reference) is Some(region) &&
          static_operation.inputs().any(input => input == WeightRegion(region)) &&
          plan.tensor(reference) is Some(tensor) else {
          abort("missing I8 fixture weight")
        }
        result.push(
          blueprint_operand(WeightInput(reference), region.byte_length(), 16L),
        )
        if tensor.scale() is Some(scale) {
          guard static_plan.weight_layout().region(scale.tensor())
            is Some(scale_region) else {
            abort("missing I8 fixture scale")
          }
          result.push(
            blueprint_operand(
              WeightScaleInput(reference, scale),
              scale_region.byte_length(),
              16L,
            ),
          )
        }
      }
      Activation(reference) =>
        result.push(
          blueprint_operand(
            ActivationInput(reference),
            full_activation_bytes(memory, reference, profile),
            16L,
          ),
        )
    }
  }
  for output in operation.outputs() {
    result.push(
      blueprint_operand(
        ActivationOutput(output),
        full_activation_bytes(memory, output, profile),
        16L,
      ),
    )
  }
  if operation.persistent_state() is KvCacheReadWrite(layer) {
    result.push(
      blueprint_operand(
        PersistentKvComponent(layer, Key),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
      ),
    )
    result.push(
      blueprint_operand(
        PersistentKvComponent(layer, Value),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
      ),
    )
  }
  if static_operation.binding().workspace_bytes() > 0L {
    result.push(
      blueprint_operand(
        Workspace,
        static_operation.binding().workspace_bytes(),
        static_operation.binding().workspace_alignment(),
      ),
    )
  }
  result
}

///|
fn i8_fixture_contracts(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  static_plan : @device_plan.StaticDevicePlan,
  memory : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  profiles : ArrayView[@launch_contract.PagedKernelProfile],
) -> @launch_contract.PagedV4LaunchContractSet raise {
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for profile in profiles {
    for index, operation in plan.operations() {
      guard resolved.bindings()[index].aot_entry_point() is Some(entry_point) else {
        abort("missing I8 fixture entry point")
      }
      inputs.push(
        @launch_contract.PagedV4AotLaunchContractInput::new(
          profile_id=profile.id(),
          operation_id=operation.id(),
          entry_point~,
          dimensions=blueprint_dimensions(),
          operands=i8_fixture_operands(
            plan,
            operation,
            static_plan.operations()[index],
            static_plan,
            memory,
            layout,
            profile,
          ),
        ),
      )
    }
  }
  @launch_contract.admit_paged_graph_v4(
    plan,
    resolved,
    layout,
    profiles,
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=4,
      max_contracts=64,
      max_operands_per_contract=32,
    ),
  )
}

///|
fn i8_fixture_artifacts(
  contracts : @launch_contract.PagedV4LaunchContractSet,
  alternate_symbols? : Bool = false,
) -> @artifact.KernelArtifactBundle raise {
  let module_digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  let identities : Array[@catalog.AotKernelEntryPoint] = []
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for contract in contracts.contracts() {
    if !identities.contains(contract.entry_point()) {
      identities.push(contract.entry_point())
      entries.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=contract.entry_point(),
          function_symbol=if alternate_symbols {
            "i8_alternate"
          } else {
            "i8_exact"
          },
        ),
      )
    }
  }
  @artifact.admit_paged_v4(
    contracts,
    [
      @artifact.KernelModuleInput::new(
        digest=module_digest,
        module_bytes=FixedArray::from_array([b'\x01', b'\x02', b'\x03']),
      ),
    ],
    entries,
    @artifact.KernelArtifactLimits::new(
      max_modules=2,
      max_entry_points=32,
      max_module_bytes=1024L,
      max_total_module_bytes=2048L,
      max_function_symbol_bytes=64,
    ),
  )
}

///|
fn i8_blueprint_fixture() -> I8BlueprintFixture raise {
  let plan = @llama.build_paged_symmetric_i8_weight_only_v1(
    blueprint_metadata(),
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
    batch=@llama.LlamaPagedBatchEnvelope::new(2),
  )
  let layout = blueprint_layout(plan)
  let target = blueprint_target(minor=9)
  let device_numeric = i8_fixture_device()
  let resolved = i8_fixture_resolved(plan, layout, target)
  let weight_layout = blueprint_weight_layout(plan)
  let static_plan = @device_plan.build(plan, weight_layout, resolved)
  let memory_plan = i8_fixture_memory(static_plan)
  let profile = blueprint_profile()
  let contracts = i8_fixture_contracts(
    plan,
    resolved,
    static_plan,
    memory_plan,
    layout,
    [profile],
  )
  let artifacts = i8_fixture_artifacts(contracts)
  let policy = @i8_admission.admit_production_llama_symmetric_i8_weight_only_v1(
    plan,
    device_numeric,
    resolved,
    @i8_admission.I8InertKernelCapabilityAdmissionLimits::new(
      max_operations=16,
      max_weights=32,
      max_canonical_bytes=65536,
    ),
  )
  let blueprint = admit_i8_paged_graph_execution_blueprint_v1(
    plan,
    device_numeric,
    policy,
    blueprint_limits(),
    layout,
    static_plan,
    memory_plan,
    profile.id(),
    contracts,
    artifacts,
  )
  {
    plan,
    device_numeric,
    policy,
    layout,
    static_plan,
    memory_plan,
    profile,
    contracts,
    artifacts,
    blueprint,
  }
}
