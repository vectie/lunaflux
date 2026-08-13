///|
fn paged_artifact_plan() -> @plan.ModelPlan raise {
  let geometry = @plan.KvCacheGeometry::new(
    layer_count=2,
    kv_head_count=2,
    head_dimension=4,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let operations : Array[@plan.PlanOperation] = [
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
  ]
  for layer_index in 0..<2 {
    let layer = @plan.KvLayerId::new(layer_index)
    let rotary_id = layer_index * 2 + 1
    let attention_id = rotary_id + 1
    let input = @plan.ActivationRef::new(rotary_id - 1)
    let rotary_output = @plan.ActivationRef::new(rotary_id)
    let attention_output = @plan.ActivationRef::new(attention_id)
    operations.push(
      @plan.PlanOperation::new_with_execution(
        id=@plan.OperationId::new(rotary_id),
        kind=RotaryEmbedding,
        inputs=[Activation(input)],
        outputs=[rotary_output],
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
    )
    operations.push(
      @plan.PlanOperation::new_with_execution(
        id=@plan.OperationId::new(attention_id),
        kind=CausalAttention,
        inputs=[Activation(rotary_output)],
        outputs=[attention_output],
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
    )
  }
  @plan.ModelPlan::new_with_kv_execution(
    identity=artifact_identity(),
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
    kv_cache_geometry=geometry,
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(4),
    kv_execution=PagedKeyValue,
  )
}

///|
fn paged_artifact_layout() -> @device_layout.DeviceKvLayout raise {
  let plan = paged_artifact_plan()
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=8192L,
    ),
  )
}

///|
fn paged_artifact_semantics(
  kind : @plan.OperationKind,
  layer : @plan.KvLayerId?,
) -> @catalog.KernelExecutionSemantics raise {
  match kind {
    RotaryEmbedding =>
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=layer,
        runtime_inputs=[StepCounts, QueryPositions],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    CausalAttention => {
      guard layer is Some(exact_layer) else { abort("missing fixture layer") }
      @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=layer,
        runtime_inputs=[
          StepCounts,
          QueryPositions,
          QueryRowOffsets,
          SequenceLengths,
          PageTableOffsets,
          PageTablePageIndices,
        ],
        persistent_state=KvCacheReadWrite(exact_layer),
        kv_layout=Some(
          @catalog.PagedKvLayoutContract::new(
            layout=LayerMajorSplitKeyValueV1,
            stable_version=1,
            tokens_per_page=4,
          ),
        ),
      )
    }
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
fn paged_artifact_catalog(
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = [
    @catalog.CatalogEntry::new_paged_v3(
      operation_kind=Embedding,
      operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
      capability=@plan.KernelCapabilityId::embedding_lookup(),
      execution_semantics=paged_artifact_semantics(Embedding, None),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(family(digest, 10)),
      workspace_bytes=0L,
      workspace_alignment=1L,
    ),
  ]
  for layer_index in 0..<2 {
    let layer = @plan.KvLayerId::new(layer_index)
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=RotaryEmbedding,
        operation_shape=RotaryShape(
          query_heads=4,
          kv_heads=2,
          head_dimension=4,
          rope_theta=10000.0,
        ),
        capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
        execution_semantics=paged_artifact_semantics(
          RotaryEmbedding,
          Some(layer),
        ),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(family(digest, 11)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=CausalAttention,
        operation_shape=AttentionShape(
          query_heads=4,
          kv_heads=2,
          head_dimension=4,
        ),
        capability=@plan.KernelCapabilityId::paged_causal_attention(),
        execution_semantics=paged_artifact_semantics(
          CausalAttention,
          Some(layer),
        ),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(family(digest, 12)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    paged_artifact_plan(),
    paged_artifact_layout(),
    @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn paged_artifact_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}

///|
fn paged_artifact_operands(
  operation_id : Int,
) -> Array[@launch_contract.AotOperand] raise {
  if operation_id == 0 {
    return [
      paged_artifact_operand(RuntimeInput(StepCounts), 20L, 4L),
      paged_artifact_operand(ExternalInput(TokenIds), 16L, 4L),
      paged_artifact_operand(WeightInput(@plan.TensorRef::new(0)), 4096L, 16L),
      paged_artifact_operand(
        ActivationOutput(@plan.ActivationRef::new(0)),
        128L,
        16L,
      ),
    ]
  }
  let layer = @plan.KvLayerId::new((operation_id - 1) / 2)
  let input = @plan.ActivationRef::new(operation_id - 1)
  let output = @plan.ActivationRef::new(operation_id)
  if operation_id % 2 == 1 {
    [
      paged_artifact_operand(RuntimeInput(StepCounts), 20L, 4L),
      paged_artifact_operand(RuntimeInput(QueryPositions), 16L, 4L),
      paged_artifact_operand(ActivationInput(input), 128L, 16L),
      paged_artifact_operand(ActivationOutput(output), 128L, 16L),
    ]
  } else {
    [
      paged_artifact_operand(RuntimeInput(StepCounts), 20L, 4L),
      paged_artifact_operand(RuntimeInput(QueryPositions), 16L, 4L),
      paged_artifact_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
      paged_artifact_operand(RuntimeInput(SequenceLengths), 8L, 4L),
      paged_artifact_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
      paged_artifact_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
      paged_artifact_operand(ActivationInput(input), 128L, 16L),
      paged_artifact_operand(ActivationOutput(output), 128L, 16L),
      paged_artifact_operand(PersistentKvComponent(layer, Key), 2048L, 256L),
      paged_artifact_operand(PersistentKvComponent(layer, Value), 2048L, 256L),
    ]
  }
}

///|
fn paged_artifact_full_contracts(
  digest : @catalog.AotArtifactDigest,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile_id = @launch_contract.KernelProfileId::from_stable_int(1)
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation_id in 0..<5 {
    let (family_id, entry_id) = if operation_id == 0 {
      (10, 3)
    } else if operation_id % 2 == 1 {
      (11, 1)
    } else {
      (12, 2)
    }
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id~,
        operation_id=@plan.OperationId::new(operation_id),
        entry_point=entry(digest, family_id, entry_id),
        dimensions=dimensions(4),
        operands=paged_artifact_operands(operation_id),
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    paged_artifact_plan(),
    paged_artifact_catalog(digest),
    paged_artifact_layout(),
    [
      @launch_contract.PagedKernelProfile::new(
        id=profile_id,
        max_query_rows=2,
        max_query_tokens=4,
        max_page_table_entries=4,
      ),
    ],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=8,
      max_operands_per_contract=16,
    ),
  )
}

///|
fn paged_artifact_contracts(
  digest : @catalog.AotArtifactDigest,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile_id = @launch_contract.KernelProfileId::from_stable_int(1)
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation_id in 1..<5 {
    let family_id = if operation_id % 2 == 1 { 11 } else { 12 }
    let entry_id = if operation_id % 2 == 1 { 1 } else { 2 }
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id~,
        operation_id=@plan.OperationId::new(operation_id),
        entry_point=entry(digest, family_id, entry_id),
        dimensions=dimensions(4),
        operands=paged_artifact_operands(operation_id),
      ),
    )
  }
  @launch_contract.admit_paged_kv(
    paged_artifact_plan(),
    paged_artifact_catalog(digest),
    paged_artifact_layout(),
    [
      @launch_contract.PagedKernelProfile::new(
        id=profile_id,
        max_query_rows=2,
        max_query_tokens=4,
        max_page_table_entries=4,
      ),
    ],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=8,
      max_operands_per_contract=16,
    ),
  )
}
