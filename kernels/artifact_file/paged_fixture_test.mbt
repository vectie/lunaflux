///|
fn paged_file_plan() -> @plan.ModelPlan {
  let layer = try! @plan.KvLayerId::new(0)
  let embedding = try! @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
  let rotary = try! @plan.PlanOperation::new_with_execution(
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
  )
  let attention = try! @plan.PlanOperation::new_with_execution(
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
  )
  try! @plan.ModelPlan::new_with_kv_execution(
    identity=fixture_identity(),
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
    operations=[embedding, rotary, attention],
    final_output=@plan.ActivationRef::new(2),
    kv_execution=PagedKeyValue,
  )
}

///|
fn paged_file_layout() -> @device_layout.DeviceKvLayout {
  let plan = paged_file_plan()
  try! @device_layout.plan(
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
fn paged_file_semantics(
  kind : @plan.OperationKind,
) -> @catalog.KernelExecutionSemantics {
  let layer = try! @plan.KvLayerId::new(0)
  match kind {
    Embedding =>
      try! @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=None,
        runtime_inputs=[StepCounts],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    RotaryEmbedding =>
      try! @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=[StepCounts, QueryPositions],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    CausalAttention =>
      try! @catalog.KernelExecutionSemantics::paged_v3(
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
            tokens_per_page=4,
          ),
        ),
      )
    _ => abort("unsupported paged file fixture")
  }
}

///|
fn paged_file_family(
  digest : @catalog.AotArtifactDigest,
  stable_id : Int,
) -> @catalog.AotKernelFamily {
  try! @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(stable_id),
  )
}

///|
fn paged_file_catalog(
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog {
  let plan = paged_file_plan()
  let entries : Array[@catalog.CatalogEntry] = []
  for operation in plan.operations() {
    let family_id = operation.id().as_int() + 10
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=paged_file_semantics(operation.kind()),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(paged_file_family(digest, family_id)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v3(),
    entries~,
  ).resolve_paged(plan, paged_file_layout(), fixture_target())
}

///|
fn paged_file_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand {
  try! @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}

///|
fn paged_file_operands(
  operation_id : Int,
) -> Array[@launch_contract.AotOperand] {
  let layout = paged_file_layout()
  let input = try! @plan.ActivationRef::new(operation_id - 1)
  let output = try! @plan.ActivationRef::new(operation_id)
  if operation_id == 1 {
    return [
      paged_file_operand(RuntimeInput(StepCounts), 20L, 4L),
      paged_file_operand(RuntimeInput(QueryPositions), 16L, 4L),
      paged_file_operand(ActivationInput(input), 128L, 16L),
      paged_file_operand(ActivationOutput(output), 128L, 16L),
    ]
  }
  let layer = try! @plan.KvLayerId::new(0)
  [
    paged_file_operand(RuntimeInput(StepCounts), 20L, 4L),
    paged_file_operand(RuntimeInput(QueryPositions), 16L, 4L),
    paged_file_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
    paged_file_operand(RuntimeInput(SequenceLengths), 8L, 4L),
    paged_file_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
    paged_file_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
    paged_file_operand(ActivationInput(input), 128L, 16L),
    paged_file_operand(ActivationOutput(output), 128L, 16L),
    paged_file_operand(
      PersistentKvComponent(layer, Key),
      layout.component_stride_bytes(),
      layout.segment_alignment_bytes(),
    ),
    paged_file_operand(
      PersistentKvComponent(layer, Value),
      layout.component_stride_bytes(),
      layout.segment_alignment_bytes(),
    ),
  ]
}

///|
fn paged_file_entry(
  digest : @catalog.AotArtifactDigest,
  operation_id : Int,
) -> @catalog.AotKernelEntryPoint {
  try! @catalog.AotKernelEntryPoint::new(
    family=paged_file_family(digest, operation_id + 10),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
      operation_id + 20,
    ),
  )
}

///|
fn paged_file_contracts() -> @launch_contract.PagedKvLaunchContractSet {
  let digest = fixture_digest()
  let profile_id = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation_id in 1..<3 {
    let resolved_operation_id = try! @plan.OperationId::new(operation_id)
    let resolved_dimensions = try! @launch_contract.AotLaunchDimensions::new(
      grid_x=4,
      grid_y=1,
      grid_z=1,
      block_x=32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    )
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id~,
        operation_id=resolved_operation_id,
        entry_point=paged_file_entry(digest, operation_id),
        dimensions=resolved_dimensions,
        operands=paged_file_operands(operation_id),
      ),
    )
  }
  try! @launch_contract.admit_paged_kv(
    paged_file_plan(),
    paged_file_catalog(digest),
    paged_file_layout(),
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
      max_profiles=1,
      max_contracts=2,
      max_operands_per_contract=12,
    ),
  )
}

///|
fn paged_valid_manifest(entry_count? : Int = 2) -> Bytes {
  let entries = [
    entry_json(11, 21, "paged_rotary"),
    entry_json(12, 22, "paged_attention"),
  ]
  manifest_json(
    [
      module_json(
        fixture_digest().sha256(),
        "kernel.cubin",
        entries[:entry_count],
      ),
    ],
    catalog_version=3,
  )
}
