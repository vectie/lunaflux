///|
fn pa_content() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn pa_family(seed : Int) -> @catalog.AotKernelFamily raise {
  let digit = (seed % 10).to_string()
  @catalog.AotKernelFamily::new(
    module_digest=@catalog.AotArtifactDigest::from_sha256(digit.repeat(64)),
    family_id=@catalog.AotKernelFamilyId::from_stable_int(seed),
  )
}

///|
fn pa_entry(seed : Int) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=pa_family(seed),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(seed),
  )
}

///|
fn pa_plan() -> @plan.ModelPlan raise {
  let layer = @plan.KvLayerId::new(0)
  let operations = [
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=16, width=4),
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
        query_heads=2,
        kv_heads=1,
        head_dimension=2,
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
      shape=AttentionShape(query_heads=2, kv_heads=1, head_dimension=2),
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
    content_digest=pa_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=2,
      max_sequence_tokens=4,
      vocabulary_size=16,
      hidden_size=4,
      intermediate_size=8,
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
    operations~,
    final_output=@plan.ActivationRef::new(2),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn pa_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = pa_plan()
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=2,
      total_page_count=4,
      max_arena_bytes=2048L,
    ),
  )
}

///|
fn pa_semantics(
  kind : @plan.OperationKind,
) -> @catalog.KernelExecutionSemantics raise {
  let layer = @plan.KvLayerId::new(0)
  match kind {
    CausalAttention =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=[
          StepCounts,
          QueryPositions,
          QueryRowOffsets,
          SequenceLengths,
          PageTableOffsets,
          PageTablePageIndices,
        ],
        persistent_state=KvCacheReadWrite(layer),
        kv_layout=Some(
          @catalog.PagedKvLayoutContract::new(
            layout=LayerMajorSplitKeyValueV1,
            stable_version=1,
            tokens_per_page=2,
          ),
        ),
      )
    RotaryEmbedding =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=[StepCounts, QueryPositions],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    _ =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=None,
        runtime_inputs=[StepCounts],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
  }
}

///|
fn pa_catalog() -> @catalog.ResolvedKernelCatalog raise {
  let plan = pa_plan()
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=pa_semantics(operation.kind()),
        compute_major=9,
        compute_minor=0,
        implementation=ContentAddressedAot(pa_family(index + 1)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    plan,
    pa_layout(),
    pa_target(),
  )
}

///|
fn pa_target(bf16? : Bool = true) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=bf16,
    supports_cublas_lt=true,
  )
}

///|
fn pa_dimensions(
  attention? : Bool = true,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=3,
    grid_y=if attention { 2 } else { 1 },
    grid_z=1,
    block_x=if attention { 1 } else { 32 },
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn pa_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn pa_rotary_operands() -> Array[@launch_contract.AotOperand] raise {
  [
    pa_operand(RuntimeInput(StepCounts), 20L, 4L),
    pa_operand(RuntimeInput(QueryPositions), 12L, 4L),
    pa_operand(ActivationInput(@plan.ActivationRef::new(0)), 48L, 16L),
    pa_operand(ActivationOutput(@plan.ActivationRef::new(1)), 48L, 16L),
  ]
}

///|
fn pa_embedding_operands() -> Array[@launch_contract.AotOperand] raise {
  [
    pa_operand(RuntimeInput(StepCounts), 20L, 4L),
    pa_operand(ExternalInput(TokenIds), 12L, 4L),
    pa_operand(WeightInput(@plan.TensorRef::new(0)), 128L, 16L),
    pa_operand(ActivationOutput(@plan.ActivationRef::new(0)), 24L, 16L),
  ]
}

///|
fn pa_attention_operands() -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  [
    pa_operand(RuntimeInput(StepCounts), 20L, 4L),
    pa_operand(RuntimeInput(QueryPositions), 12L, 4L),
    pa_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
    pa_operand(RuntimeInput(SequenceLengths), 8L, 4L),
    pa_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
    pa_operand(RuntimeInput(PageTablePageIndices), 12L, 4L),
    pa_operand(ActivationInput(@plan.ActivationRef::new(1)), 48L, 16L),
    pa_operand(ActivationOutput(@plan.ActivationRef::new(2)), 24L, 16L),
    pa_operand(PersistentKvComponent(layer, Key), 1024L, 256L),
    pa_operand(PersistentKvComponent(layer, Value), 1024L, 256L),
  ]
}

///|
fn pa_contracts(
  exact_dimensions? : Bool = true,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=3,
    max_page_table_entries=3,
  )
  let inputs = [
    @launch_contract.PagedKvAotLaunchContractInput::new(
      profile_id=profile.id(),
      operation_id=@plan.OperationId::new(0),
      entry_point=pa_entry(1),
      dimensions=pa_dimensions(attention=false),
      operands=pa_embedding_operands(),
    ),
    @launch_contract.PagedKvAotLaunchContractInput::new(
      profile_id=profile.id(),
      operation_id=@plan.OperationId::new(1),
      entry_point=pa_entry(2),
      dimensions=pa_dimensions(attention=false),
      operands=pa_rotary_operands(),
    ),
    @launch_contract.PagedKvAotLaunchContractInput::new(
      profile_id=profile.id(),
      operation_id=@plan.OperationId::new(2),
      entry_point=pa_entry(3),
      dimensions=if exact_dimensions {
        pa_dimensions()
      } else {
        pa_dimensions(attention=false)
      },
      operands=pa_attention_operands(),
    ),
  ]
  @launch_contract.admit_paged_graph(
    pa_plan(),
    pa_catalog(),
    pa_layout(),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=4,
      max_operands_per_contract=12,
    ),
  )
}

///|
fn pa_attention_contract(
  exact_dimensions? : Bool = true,
) -> @launch_contract.PagedKvAotLaunchContract raise {
  pa_contracts(exact_dimensions~).contracts()[2]
}

///|
fn pa_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="1111111111111111111111111111111111111111111111111111111111111111",
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate~,
    max_registers=128,
  )
}

///|
fn pa_limits() -> @luna_cuda_paged_attention_aot.PagedAttentionAotLimits raise {
  @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
    max_source_bytes=32768,
    max_recipe_bytes=4096,
  )
}

///|
fn pa_lowering() -> @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering raise {
  let operation = pa_plan().operations()[2]
  let contracts = pa_contracts()
  let contract = contracts.contracts()[2]
  @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
    operation,
    contract.profile(),
    pa_layout(),
    contract.entry_point().entry_point_id(),
    pa_target(),
    pa_compiler(),
    contract.operands(),
    pa_limits(),
  )
}
