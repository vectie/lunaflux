///|
fn d2_v4_content(
  digest? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(digest)
}

///|
fn d2_v4_operations() -> Array[@plan.PlanOperation] {
  let layer = try! @plan.KvLayerId::new(0)
  [
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=8, width=1),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(1),
      kind=OutputProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=ProjectionShape(input_width=1, output_width=1),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(2),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(1))],
      outputs=[@plan.ActivationRef::new(2)],
      shape=RotaryShape(
        query_heads=1,
        kv_heads=1,
        head_dimension=1,
        rope_theta=10000.0,
      ),
      required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[StepCounts, QueryPositions],
      persistent_state=NoPersistentState,
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(3),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(2))],
      outputs=[@plan.ActivationRef::new(3)],
      shape=AttentionShape(query_heads=1, kv_heads=1, head_dimension=1),
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
}

///|
fn d2_v4_plan(
  mode : Int,
  content_digest? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @plan.ModelPlan {
  d2_v4_plan_with_content(mode, d2_v4_content(digest=content_digest))
}

///|
fn d2_v4_plan_with_content(
  mode : Int,
  content : @spec.ContentDigest,
) -> @plan.ModelPlan {
  let operations = d2_v4_operations()
  let selected = [try! @plan.OperationId::new(1)]
  let numeric_schema = match mode {
    1 => @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
    2 =>
      @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, selected)
    3 =>
      @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
        operations, selected,
      )
    _ => abort("unknown D2 v4 numeric mode")
  }
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=content,
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
      vocabulary_size=8,
      hidden_size=1,
      intermediate_size=2,
      attention_head_count=1,
      kv_head_count=1,
      head_dimension=1,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=1,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(3),
    kv_execution=PagedKeyValue,
    numeric_schema~,
  )
}

///|
fn d2_v4_operation_digest(
  plan : @plan.ModelPlan,
  index : Int,
) -> @numeric_contract.OperationExecutionDigest {
  plan.numeric_binding().operations()[index].execution().digest()
}

///|
fn d2_v4_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics {
  let layer = try! @plan.KvLayerId::new(0)
  match operation.kind() {
    RotaryEmbedding =>
      try! @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=operation.runtime_inputs(),
        persistent_state=operation.persistent_state(),
        kv_layout=None,
      )
    CausalAttention =>
      try! @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=Some(layer),
        runtime_inputs=operation.runtime_inputs(),
        persistent_state=operation.persistent_state(),
        kv_layout=Some(
          @catalog.PagedKvLayoutContract::new(
            layout=LayerMajorSplitKeyValueV1,
            stable_version=1,
            tokens_per_page=4,
          ),
        ),
      )
    _ =>
      try! @catalog.KernelExecutionSemantics::paged_v3(
        kv_layer=None,
        runtime_inputs=operation.runtime_inputs(),
        persistent_state=operation.persistent_state(),
        kv_layout=None,
      )
  }
}

///|
fn d2_v4_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn d2_v4_kv_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
  try! @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=4,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn d2_v4_catalog(plan : @plan.ModelPlan) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  let module_digest = try! @catalog.AotArtifactDigest::from_sha256(
    "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace",
  )
  for index, operation in plan.operations() {
    let family = @catalog.AotKernelFamily::new(
      module_digest~,
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(index + 1),
    )
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=d2_v4_semantics(operation),
        operation_execution_digest=d2_v4_operation_digest(plan, index),
        aot_entry_point=@catalog.AotKernelEntryPoint::new(
          family~,
          entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
            index + 101,
          ),
        ),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(family),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  ).resolve_paged_v4(plan, d2_v4_kv_layout(plan), d2_v4_target())
}
