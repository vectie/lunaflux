///|
fn bundle_fixture_module(operation : Int) -> Bytes {
  if operation == 1 {
    b"cubin-b"
  } else {
    b"cubin-a"
  }
}

///|
fn bundle_fixture_digest(operation : Int) -> @catalog.AotArtifactDigest raise {
  @catalog.AotArtifactDigest::from_sha256(
    kernel_bundle_sha256(bundle_fixture_module(operation)),
  )
}

///|
fn bundle_fixture_entry(operation : Int) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=bundle_fixture_digest(operation),
      family_id=@catalog.AotKernelFamilyId::from_stable_int(operation + 11),
    ),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
      operation + 1,
    ),
  )
}

///|
fn bundle_fixture_plan() -> @plan.ModelPlan raise {
  let layer = @plan.KvLayerId::new(0)
  let operations = [
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(1),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(0))],
      outputs=[@plan.ActivationRef::new(1)],
      shape=RotaryShape(
        query_heads=4,
        kv_heads=2,
        head_dimension=4,
        rope_theta=10000.0,
      ),
      required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[StepCounts, QueryPositions],
      persistent_state=NoPersistentState,
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(2),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(1))],
      outputs=[@plan.ActivationRef::new(2)],
      shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
      required_capability=@plan.KernelCapabilityId::paged_causal_attention(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[
        StepCounts,
        QueryPositions,
        QueryRowOffsets,
        SequenceLengths,
        PageTableOffsets,
        PageTablePageIndices,
      ],
      persistent_state=KvCacheReadWrite(layer),
    ),
  ]
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256("a".repeat(64)),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=2,
      max_sequence_tokens=8,
      vocabulary_size=128,
      hidden_size=16,
      intermediate_size=32,
      attention_head_count=4,
      kv_head_count=2,
      head_dimension=4,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(2),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn bundle_fixture_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn bundle_fixture_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = bundle_fixture_plan()
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn bundle_fixture_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics raise {
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout=if operation.kind() == CausalAttention {
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page=4,
        ),
      )
    } else {
      None
    },
  )
}

///|
fn bundle_fixture_catalog() -> @catalog.ResolvedKernelCatalog raise {
  let plan = bundle_fixture_plan()
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=bundle_fixture_semantics(operation),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(bundle_fixture_entry(index).family()),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    plan,
    bundle_fixture_layout(),
    bundle_fixture_target(),
  )
}

///|
fn bundle_fixture_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn bundle_fixture_operands(
  operation : Int,
) -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  match operation {
    0 =>
      [
        bundle_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
        bundle_fixture_operand(ExternalInput(TokenIds), 16L, 4L),
        bundle_fixture_operand(WeightInput(@plan.TensorRef::new(0)), 4096L, 16L),
        bundle_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(0)),
          128L,
          16L,
        ),
      ]
    1 =>
      [
        bundle_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
        bundle_fixture_operand(RuntimeInput(QueryPositions), 16L, 4L),
        bundle_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(0)),
          128L,
          16L,
        ),
        bundle_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(1)),
          128L,
          16L,
        ),
      ]
    _ =>
      [
        bundle_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
        bundle_fixture_operand(RuntimeInput(QueryPositions), 16L, 4L),
        bundle_fixture_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
        bundle_fixture_operand(RuntimeInput(SequenceLengths), 8L, 4L),
        bundle_fixture_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
        bundle_fixture_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
        bundle_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(1)),
          256L,
          16L,
        ),
        bundle_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(2)),
          128L,
          16L,
        ),
        bundle_fixture_operand(PersistentKvComponent(layer, Key), 2048L, 256L),
        bundle_fixture_operand(PersistentKvComponent(layer, Value), 2048L, 256L),
      ]
  }
}

///|
fn bundle_fixture_dimensions(
  operation : Int,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=if operation == 2 { 4 } else { operation + 1 },
    grid_y=if operation == 2 { 4 } else { 1 },
    grid_z=1,
    block_x=if operation == 2 { 32 } else { 256 },
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn bundle_fixture_contracts() -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation in 0..<3 {
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=@plan.OperationId::new(operation),
        entry_point=bundle_fixture_entry(operation),
        dimensions=bundle_fixture_dimensions(operation),
        operands=bundle_fixture_operands(operation),
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    bundle_fixture_plan(),
    bundle_fixture_catalog(),
    bundle_fixture_layout(),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=3,
      max_operands_per_contract=16,
    ),
  )
}
