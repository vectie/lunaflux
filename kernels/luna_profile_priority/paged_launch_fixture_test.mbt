///|
fn paged_profile_device_layout(
  plan : @plan.ModelPlan,
  tokens_per_page? : Int = 4,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page~,
      total_page_count=32,
      max_arena_bytes=65536L,
    ),
  )
}

///|
fn paged_profile_family(index : Int) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest=@catalog.AotArtifactDigest::from_sha256(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    family_id=@catalog.AotKernelFamilyId::from_stable_int(index + 1),
  )
}

///|
fn paged_profile_execution_semantics(
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
) -> @catalog.KernelExecutionSemantics raise {
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout=match operation.persistent_state() {
      NoPersistentState => None
      KvCacheReadWrite(_) =>
        Some(
          @catalog.PagedKvLayoutContract::new(
            layout=layout.geometry().layout(),
            stable_version=layout.stable_layout_version(),
            tokens_per_page=layout.tokens_per_page(),
          ),
        )
    },
  )
}

///|
fn paged_profile_same_catalog_semantics(
  left : @plan.PlanOperation,
  right : @plan.PlanOperation,
) -> Bool {
  left.kind() == right.kind() &&
  left.shape() == right.shape() &&
  left.required_capability() == right.required_capability() &&
  left.kv_layer() == right.kv_layer() &&
  left.runtime_inputs() == right.runtime_inputs() &&
  left.persistent_state() == right.persistent_state()
}

///|
fn paged_profile_family_index(plan : @plan.ModelPlan, index : Int) -> Int {
  let operation = plan.operations()[index]
  for earlier in 0..<index {
    if paged_profile_same_catalog_semantics(
        plan.operations()[earlier],
        operation,
      ) {
      return earlier
    }
  }
  index
}

///|
fn paged_profile_resolved_catalog(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    if paged_profile_family_index(plan, index) != index {
      continue
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=paged_profile_execution_semantics(operation, layout),
        compute_major=target.compute_major(),
        compute_minor=target.compute_minor(),
        implementation=ContentAddressedAot(paged_profile_family(index)),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() == 0L {
          1L
        } else {
          16L
        },
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    plan, layout, target,
  )
}

///|
fn paged_profile_runtime_bytes(
  role : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match role {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn paged_profile_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn paged_profile_operands(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
) -> Array[@launch_contract.AotOperand] raise {
  let operands : Array[@launch_contract.AotOperand] = []
  for role in operation.runtime_inputs() {
    operands.push(
      paged_profile_operand(
        RuntimeInput(role),
        paged_profile_runtime_bytes(role, profile),
        4L,
      ),
    )
  }
  for input in operation.inputs() {
    operands.push(
      paged_profile_operand(
        match input {
          ModelInput(value) => ExternalInput(value)
          Tensor(value) => WeightInput(value)
          Activation(value) => ActivationInput(value)
        },
        1L,
        1L,
      ),
    )
  }
  for output in operation.outputs() {
    operands.push(paged_profile_operand(ActivationOutput(output), 1L, 1L))
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      operands.push(
        paged_profile_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
      operands.push(
        paged_profile_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
    }
  }
  operands
}

///|
fn paged_profile_launch_inputs(
  plan : @plan.ModelPlan,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
) -> Array[@launch_contract.PagedKvAotLaunchContractInput] raise {
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for index, operation in plan.operations() {
    if operation.kind() is (RotaryEmbedding | CausalAttention) {
      inputs.push(
        @launch_contract.PagedKvAotLaunchContractInput::new(
          profile_id=profile.id(),
          operation_id=operation.id(),
          entry_point=@catalog.AotKernelEntryPoint::new(
            family=paged_profile_family(paged_profile_family_index(plan, index)),
            entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(1),
          ),
          dimensions=@launch_contract.AotLaunchDimensions::new(
            grid_x=1,
            grid_y=1,
            grid_z=1,
            block_x=32,
            block_y=1,
            block_z=1,
            shared_memory_bytes=0,
          ),
          operands=paged_profile_operands(operation, profile, layout),
        ),
      )
    }
  }
  inputs
}

///|
fn paged_profile_launches_with(
  plan : @plan.ModelPlan,
  profile : @launch_contract.PagedKernelProfile,
  tokens_per_page : Int,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let layout = paged_profile_device_layout(plan, tokens_per_page~)
  @launch_contract.admit_paged_kv(
    plan,
    paged_profile_resolved_catalog(plan, layout, profile_test_target()),
    layout,
    [profile],
    paged_profile_launch_inputs(plan, profile, layout),
    @launch_contract.LaunchContractLimits::new(
      max_profiles=4,
      max_contracts=32,
      max_operands_per_contract=16,
    ),
  )
}

///|
fn paged_profile_launches(
  plan : @plan.ModelPlan,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  paged_profile_launches_with(
    plan,
    @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(7),
      max_query_rows=4,
      max_query_tokens=16,
      max_page_table_entries=16,
    ),
    4,
  )
}
