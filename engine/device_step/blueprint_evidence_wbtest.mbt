///|
fn blueprint_resolved_catalog(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  rotary_workspace? : Int64 = 64L,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  for operation in plan.operations() {
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
    let semantics = @catalog.KernelExecutionSemantics::paged_v2(
      kv_layer=operation.kv_layer(),
      runtime_inputs=operation.runtime_inputs(),
      persistent_state=operation.persistent_state(),
      kv_layout~,
    )
    let family = @catalog.AotKernelFamily::new(
      module_digest=digest,
      family_id=@catalog.AotKernelFamilyId::from_stable_int(
        operation.id().as_int() + 1,
      ),
    )
    let workspace = if operation.kind() == RotaryEmbedding {
      rotary_workspace
    } else {
      0L
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v2(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=semantics,
        compute_major=8,
        compute_minor=target.compute_minor(),
        implementation=ContentAddressedAot(family),
        workspace_bytes=workspace,
        workspace_alignment=if workspace > 0L { 16L } else { 1L },
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v2(), entries~).resolve_paged(
    plan, layout, target,
  )
}

///|
fn blueprint_memory_plan(
  metadata : @spec.LlamaModelMetadata,
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
) -> @device_memory.DeviceMemoryPlan raise {
  let mut rotary_workspace = 0L
  for operation in plan.operations() {
    if operation.kind() == RotaryEmbedding {
      rotary_workspace = operation.workspace_bytes()
    }
  }
  let static_plan = @device_plan.build(
    plan,
    blueprint_weight_layout(metadata),
    blueprint_resolved_catalog(plan, layout, target, rotary_workspace~),
  )
  @device_memory.plan(
    static_plan,
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=8192L,
    ),
  )
}

///|
fn blueprint_profile(
  id? : Int = 1,
) -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(id),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
}

///|
fn blueprint_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn blueprint_contract_operands(
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
  rotary_input_bytes? : Int64 = 64L,
  rotary_input_alignment? : Int64 = 16L,
  rotary_workspace? : Int64 = 64L,
) -> Array[@launch_contract.AotOperand] raise {
  guard operation.inputs()[0] is Activation(input) &&
    operation.outputs()[0] is output else {
    abort("blueprint fixture requires activation operation")
  }
  match operation.kind() {
    RotaryEmbedding =>
      [
        blueprint_operand(RuntimeInput(StepCounts), 20L, 4L),
        blueprint_operand(RuntimeInput(QueryPositions), 16L, 4L),
        blueprint_operand(
          ActivationInput(input),
          rotary_input_bytes,
          rotary_input_alignment,
        ),
        blueprint_operand(ActivationOutput(output), 64L, 16L),
        blueprint_operand(Workspace, rotary_workspace, 16L),
      ]
    CausalAttention => {
      guard operation.kv_layer() is Some(layer) else {
        abort("blueprint fixture requires KV layer")
      }
      [
        blueprint_operand(RuntimeInput(StepCounts), 20L, 4L),
        blueprint_operand(RuntimeInput(QueryPositions), 16L, 4L),
        blueprint_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
        blueprint_operand(RuntimeInput(SequenceLengths), 8L, 4L),
        blueprint_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
        blueprint_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
        blueprint_operand(ActivationInput(input), 64L, 16L),
        blueprint_operand(ActivationOutput(output), 32L, 16L),
        blueprint_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
        blueprint_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      ]
    }
    _ => abort("blueprint fixture requires paged operation")
  }
}

///|
fn blueprint_dimensions() -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn blueprint_contract_set(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  entry_bias? : Int = 0,
  rotary_input_bytes? : Int64 = 64L,
  rotary_input_alignment? : Int64 = 16L,
  rotary_workspace? : Int64 = 64L,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  let profile = blueprint_profile()
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  for operation in plan.operations() {
    if operation.kind() is (RotaryEmbedding | CausalAttention) {
      let family = @catalog.AotKernelFamily::new(
        module_digest=digest,
        family_id=@catalog.AotKernelFamilyId::from_stable_int(
          operation.id().as_int() + 1,
        ),
      )
      inputs.push(
        @launch_contract.PagedKvAotLaunchContractInput::new(
          profile_id=profile.id(),
          operation_id=operation.id(),
          entry_point=@catalog.AotKernelEntryPoint::new(
            family~,
            entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
              operation.id().as_int() + entry_bias + 1,
            ),
          ),
          dimensions=blueprint_dimensions(),
          operands=blueprint_contract_operands(
            operation,
            layout,
            rotary_input_bytes~,
            rotary_input_alignment~,
            rotary_workspace~,
          ),
        ),
      )
    }
  }
  @launch_contract.admit_paged_kv(
    plan,
    blueprint_resolved_catalog(plan, layout, target, rotary_workspace~),
    layout,
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=4,
      max_contracts=16,
      max_operands_per_contract=16,
    ),
  )
}

///|
fn blueprint_artifacts(
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> @artifact.KernelArtifactBundle raise {
  let entries : Array[@artifact.KernelEntryPointInput] = []
  let identities : Array[@catalog.AotKernelEntryPoint] = []
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  for contract in contracts.contracts() {
    if !identities.contains(contract.entry_point()) {
      identities.push(contract.entry_point())
      entries.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=contract.entry_point(),
          function_symbol="paged_exact",
        ),
      )
    }
  }
  @artifact.admit_paged_kv(
    contracts,
    [
      @artifact.KernelModuleInput::new(
        digest~,
        module_bytes=FixedArray::from_array([b'\x01', b'\x02', b'\x03']),
      ),
    ],
    entries,
    @artifact.KernelArtifactLimits::new(
      max_modules=2,
      max_entry_points=8,
      max_module_bytes=1024L,
      max_total_module_bytes=2048L,
      max_function_symbol_bytes=64,
    ),
  )
}

///|
fn blueprint_limits(max_tokens? : Int = 4) -> DeviceStepLimits raise {
  DeviceStepLimits::new(
    max_rows=2,
    max_tokens~,
    max_pages=4,
    max_sequence_tokens=8,
    max_token_id=7,
  )
}

///|
fn blueprint_v1_artifacts(
  identity : @spec.ModelIdentity,
  target : @catalog.DeviceTarget,
) -> @artifact.KernelArtifactBundle raise {
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let model = @plan.ModelPlan::new_stateless(
    identity~,
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=1,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[operation],
    final_output=@plan.ActivationRef::new(0),
  )
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  let family = @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(99),
  )
  let resolved = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries=[
      @catalog.CatalogEntry::new(
        semantic_version=@catalog.KernelSemanticVersion::v1(),
        operation_kind=Embedding,
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        compute_major=target.compute_major(),
        compute_minor=target.compute_minor(),
        implementation=ContentAddressedAot(family),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    ],
  ).resolve(model, target)
  let profile = @launch_contract.KernelProfile::full_prefill(
    id=@launch_contract.KernelProfileId::from_stable_int(99),
    batch_rows=1,
    sequence_tokens=1,
    token_rows=1L,
  )
  let entry = @catalog.AotKernelEntryPoint::new(
    family~,
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(99),
  )
  let contracts = @launch_contract.admit(
    model,
    resolved,
    [profile],
    [
      @launch_contract.AotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=entry,
        dimensions=blueprint_dimensions(),
        operands=[
          blueprint_operand(ExternalInput(TokenIds), 4L, 4L),
          blueprint_operand(WeightInput(@plan.TensorRef::new(0)), 64L, 16L),
          blueprint_operand(
            ActivationOutput(@plan.ActivationRef::new(0)),
            8L,
            16L,
          ),
        ],
      ),
    ],
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=2,
      max_operands_per_contract=4,
    ),
  )
  @artifact.admit(
    contracts,
    [
      @artifact.KernelModuleInput::new(
        digest~,
        module_bytes=FixedArray::from_array([b'\x01', b'\x02', b'\x03']),
      ),
    ],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point=entry,
        function_symbol="legacy_exact",
      ),
    ],
    @artifact.KernelArtifactLimits::new(
      max_modules=2,
      max_entry_points=2,
      max_module_bytes=1024L,
      max_total_module_bytes=2048L,
      max_function_symbol_bytes=64,
    ),
  )
}
