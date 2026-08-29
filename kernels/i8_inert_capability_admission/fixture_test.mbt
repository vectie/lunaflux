///|
fn i8_admission_content(seed : String) -> @spec.ContentDigest {
  let content = if seed == "base" {
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  } else {
    "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
  try! @spec.ContentDigest::from_sha256(content)
}

///|
fn i8_admission_constraints() -> @plan.ShapeConstraints {
  try! @plan.ShapeConstraints::new(
    max_batch_rows=2,
    max_sequence_tokens=64,
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=4,
  )
}

///|
fn i8_admission_geometry() -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=2,
    head_dimension=4,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
fn i8_admission_operations() -> Array[@plan.PlanOperation] {
  let layer = try! @plan.KvLayerId::new(0)
  [
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(1),
      kind=GatedMlp,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
        Tensor(@plan.TensorRef::new(3)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=GatedMlpShape(hidden_size=16, intermediate_size=32),
      required_capability=@plan.KernelCapabilityId::gated_mlp(),
      workspace_bytes=16L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(2),
      kind=OutputProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(1)),
        Tensor(@plan.TensorRef::new(4)),
      ],
      outputs=[@plan.ActivationRef::new(2)],
      shape=ProjectionShape(input_width=16, output_width=16),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(3),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(2))],
      outputs=[@plan.ActivationRef::new(3)],
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
    try! @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(4),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(3))],
      outputs=[@plan.ActivationRef::new(4)],
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
}

///|
fn i8_admission_plan(
  mode : Int,
  identity_seed? : String = "base",
) -> @plan.ModelPlan {
  let operations = i8_admission_operations()
  let gated_id = try! @plan.OperationId::new(1)
  let projection_id = try! @plan.OperationId::new(2)
  let selected = match mode {
    1 => [gated_id, projection_id]
    2 => [gated_id]
    3 => []
    _ => abort("unknown I8 admission fixture mode")
  }
  let numeric_schema = if mode == 3 {
    @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
  } else {
    @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
      operations, selected,
    )
  }
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=i8_admission_content(identity_seed),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=i8_admission_constraints(),
    kv_cache_geometry=i8_admission_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations~,
    final_output=@plan.ActivationRef::new(4),
    kv_execution=PagedKeyValue,
    numeric_schema~,
  )
}

///|
fn i8_admission_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
  try! @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=16384L,
    ),
  )
}

///|
fn i8_admission_device(
  major? : Int = 8,
  minor? : Int = 9,
  cublas_lt? : Bool = true,
) -> @device.DeviceNumericCapability {
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=major,
    compute_minor=minor,
    total_memory_bytes=8L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=cublas_lt,
  )
  try! @device.admit_symmetric_i8_weight_only_per_output_channel_v1(observed)
}

///|
fn i8_admission_fp8_device() -> @device.DeviceNumericCapability {
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=8,
    compute_minor=9,
    total_memory_bytes=8L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try! @device.admit_finite_fp8_e4m3_w8a8(observed)
}

///|
fn i8_admission_target(
  major? : Int = 8,
  minor? : Int = 9,
  cublas_lt? : Bool = true,
) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=major,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=cublas_lt,
  )
}

///|
fn i8_admission_operation_digest(
  plan : @plan.ModelPlan,
  index : Int,
) -> @numeric_contract.OperationExecutionDigest {
  plan.numeric_binding().operations()[index].execution().digest()
}

///|
fn i8_admission_family(
  module_digest : String,
  id : Int,
) -> @catalog.AotKernelFamily {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(module_digest)
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(id)
  @catalog.AotKernelFamily::new(module_digest=digest, family_id~)
}

///|
fn i8_admission_semantics(
  plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics {
  let kv_layout = if operation.kind() is CausalAttention {
    Some(
      try! @catalog.PagedKvLayoutContract::new(
        layout=plan.kv_cache_geometry().layout(),
        stable_version=plan.kv_cache_geometry().layout().stable_version(),
        tokens_per_page=4,
      ),
    )
  } else {
    None
  }
  try! @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  )
}

///|
fn i8_admission_entry(
  plan : @plan.ModelPlan,
  index : Int,
  module_digest : String,
  entry_offset? : Int = 0,
) -> @catalog.CatalogEntry {
  let operation = plan.operations()[index]
  let family = i8_admission_family(module_digest, index + 1)
  try! @catalog.CatalogEntry::new_paged_v4(
    operation_kind=operation.kind(),
    operation_shape=operation.shape(),
    capability=operation.required_capability(),
    execution_semantics=i8_admission_semantics(plan, operation),
    operation_execution_digest=i8_admission_operation_digest(plan, index),
    aot_entry_point=@catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
        101 + index + entry_offset,
      ),
    ),
    compute_major=8,
    compute_minor=9,
    implementation=ContentAddressedAot(family),
    workspace_bytes=if index == 1 { 32L } else { 0L },
    workspace_alignment=if index == 1 { 16L } else { 1L },
  )
}

///|
fn i8_admission_resolved(
  plan : @plan.ModelPlan,
  module_digest? : String = "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace",
  entry_offset? : Int = 0,
  reverse_catalog? : Bool = false,
  cublas_lt? : Bool = true,
) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  if reverse_catalog {
    for index = plan.operations().length() - 1; index >= 0; index = index - 1 {
      entries.push(
        i8_admission_entry(plan, index, module_digest, entry_offset~),
      )
    }
  } else {
    for index in 0..<plan.operations().length() {
      entries.push(
        i8_admission_entry(plan, index, module_digest, entry_offset~),
      )
    }
  }
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  )
  try! catalog.resolve_paged_v4(
    plan,
    i8_admission_layout(plan),
    i8_admission_target(cublas_lt~),
  )
}

///|
fn i8_admission_legacy_resolved(
  plan : @plan.ModelPlan,
) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    let family = i8_admission_family(
      "2468ace013579bdf2468ace013579bdf2468ace013579bdf2468ace013579bdf",
      index + 1,
    )
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=i8_admission_semantics(plan, operation),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(family),
        workspace_bytes=if index == 1 { 32L } else { 0L },
        workspace_alignment=if index == 1 { 16L } else { 1L },
      ),
    )
  }
  let catalog = try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v3(),
    entries~,
  )
  try! catalog.resolve_paged(
    plan,
    i8_admission_layout(plan),
    i8_admission_target(),
  )
}

///|
fn i8_admission_limits(
  max_operations? : Int = 8,
  max_weights? : Int = 16,
  max_canonical_bytes? : Int = 16384,
) -> @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits {
  try! @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits::new(
    max_operations~,
    max_weights~,
    max_canonical_bytes~,
  )
}

///|
fn i8_admit_fixture(
  plan : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  device? : @device.DeviceNumericCapability = i8_admission_device(),
  limits? : @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits = i8_admission_limits(),
) -> @i8_inert_capability_admission.I8InertKernelCapabilityAdmission {
  try! @i8_inert_capability_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
    plan, device, resolved, limits,
  )
}
