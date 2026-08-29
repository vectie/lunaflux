///|
fn fp8_bundle_operations() -> Array[@plan.PlanOperation] {
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
      workspace_bytes=4L,
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
fn fp8_bundle_plan() -> @plan.ModelPlan {
  let operations = fp8_bundle_operations()
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=@spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4L),
    operations~,
    final_output=@plan.ActivationRef::new(3),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
      operations,
      [try! @plan.OperationId::new(1)],
    ),
  )
}

///|
fn fp8_bundle_layout(plan : @plan.ModelPlan) -> @device_layout.DeviceKvLayout {
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
fn fp8_bundle_profile() -> @launch_contract.PagedKernelProfile {
  try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=4,
    max_page_table_entries=2,
  )
}

///|
fn fp8_bundle_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn fp8_bundle_bytes() -> Bytes {
  b"real-fp8-offline-cubin"
}

///|
fn fp8_bundle_family(index : Int) -> @catalog.AotKernelFamily {
  @catalog.AotKernelFamily::new(
    module_digest=try! @catalog.AotArtifactDigest::from_sha256(
      kernel_bundle_sha256(fp8_bundle_bytes()),
    ),
    family_id=try! @catalog.AotKernelFamilyId::from_stable_int(index + 1),
  )
}

///|
fn fp8_bundle_entry(index : Int) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=fp8_bundle_family(index),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(
      if index == 1 {
        1
      } else {
        index + 1
      },
    ),
  )
}

///|
fn fp8_bundle_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics {
  let layer = try! @plan.KvLayerId::new(0)
  try! @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=if operation.kind() is (RotaryEmbedding | CausalAttention) {
      Some(layer)
    } else {
      None
    },
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout=if operation.kind() == CausalAttention {
      Some(
        try! @catalog.PagedKvLayoutContract::new(
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
fn fp8_bundle_resolved(
  plan : @plan.ModelPlan,
) -> @catalog.ResolvedKernelCatalog {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    let family = fp8_bundle_family(index)
    entries.push(
      try! @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=fp8_bundle_semantics(operation),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=fp8_bundle_entry(index),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(family),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() == 0L {
          1L
        } else {
          4L
        },
      ),
    )
  }
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  ).resolve_paged_v4(plan, fp8_bundle_layout(plan), fp8_bundle_target())
}

///|
fn fp8_bundle_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand {
  try! @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn fp8_bundle_operands(
  plan : @plan.ModelPlan,
  index : Int,
) -> Array[@launch_contract.AotOperand] {
  let layer = try! @plan.KvLayerId::new(0)
  match index {
    0 =>
      [
        fp8_bundle_operand(RuntimeInput(StepCounts), 20L, 4L),
        fp8_bundle_operand(ExternalInput(TokenIds), 16L, 4L),
        fp8_bundle_operand(WeightInput(try! @plan.TensorRef::new(0)), 16L, 16L),
        fp8_bundle_operand(
          ActivationOutput(try! @plan.ActivationRef::new(0)),
          8L,
          16L,
        ),
      ]
    1 => {
      let admitted = try! @luna_cuda_fp8_projection_aot.derive_fp8_projection_operands(
        plan,
        operation=try! @plan.OperationId::new(1),
        fp8_bundle_profile(),
      )
      let result : Array[@launch_contract.AotOperand] = []
      for operand in admitted {
        result.push(operand)
      }
      result
    }
    2 =>
      [
        fp8_bundle_operand(RuntimeInput(StepCounts), 20L, 4L),
        fp8_bundle_operand(RuntimeInput(QueryPositions), 16L, 4L),
        fp8_bundle_operand(
          ActivationInput(try! @plan.ActivationRef::new(1)),
          24L,
          16L,
        ),
        fp8_bundle_operand(
          ActivationOutput(try! @plan.ActivationRef::new(2)),
          24L,
          16L,
        ),
      ]
    3 => {
      let layout = fp8_bundle_layout(plan)
      [
        fp8_bundle_operand(RuntimeInput(StepCounts), 20L, 4L),
        fp8_bundle_operand(RuntimeInput(QueryPositions), 16L, 4L),
        fp8_bundle_operand(RuntimeInput(QueryRowOffsets), 8L, 4L),
        fp8_bundle_operand(RuntimeInput(SequenceLengths), 4L, 4L),
        fp8_bundle_operand(RuntimeInput(PageTableOffsets), 8L, 4L),
        fp8_bundle_operand(RuntimeInput(PageTablePageIndices), 8L, 4L),
        fp8_bundle_operand(
          ActivationInput(try! @plan.ActivationRef::new(2)),
          24L,
          16L,
        ),
        fp8_bundle_operand(
          ActivationOutput(try! @plan.ActivationRef::new(3)),
          8L,
          16L,
        ),
        fp8_bundle_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
        fp8_bundle_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      ]
    }
    _ => abort("unknown FP8 bundle fixture operation")
  }
}

///|
fn fp8_bundle_contracts(
  plan : @plan.ModelPlan,
) -> @launch_contract.PagedV4LaunchContractSet {
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for index in 0..<4 {
    inputs.push(
      try! @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=fp8_bundle_profile().id(),
        operation_id=@plan.OperationId::new(index),
        entry_point=fp8_bundle_entry(index),
        dimensions=@launch_contract.AotLaunchDimensions::new(
          grid_x=1,
          grid_y=1,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        operands=fp8_bundle_operands(plan, index),
      ),
    )
  }
  try! @launch_contract.admit_paged_graph_v4(
    plan,
    fp8_bundle_resolved(plan),
    fp8_bundle_layout(plan),
    [fp8_bundle_profile()],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=8,
      max_operands_per_contract=20,
    ),
  )
}

///|
fn fp8_bundle_compiler() -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="3".repeat(64),
      major=12,
      minor=8,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn fp8_bundle_candidate(
  plan : @plan.ModelPlan,
) -> @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLowering {
  try! @luna_cuda_fp8_projection_aot.lower_fp8_projection_cuda_aot_candidate(
    plan,
    operation=@plan.OperationId::new(1),
    fp8_bundle_profile(),
    fp8_bundle_target(),
    fp8_bundle_compiler(),
    fp8_bundle_operands(plan, 1),
    @luna_cuda_fp8_projection_aot.Fp8ProjectionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
}

///|
fn fp8_bundle_artifacts(
  plan : @plan.ModelPlan,
  output_symbol? : String = "lunaflux_luna_dense_projection_fp8_e4m3_w8a8_release_v1",
) -> @artifact.KernelArtifactBundle {
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for index in 0..<4 {
    entries.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=fp8_bundle_entry(index),
        function_symbol=if index == 1 {
          output_symbol
        } else {
          "fixture_kernel_" + index.to_string()
        },
      ),
    )
  }
  try! @artifact.admit_paged_v4(
    fp8_bundle_contracts(plan),
    [
      @artifact.KernelModuleInput::from_bytes(
        digest=fp8_bundle_family(0).module_digest(),
        module_bytes=fp8_bundle_bytes(),
      ),
    ],
    entries,
    @artifact.KernelArtifactLimits::new(
      max_modules=2,
      max_entry_points=8,
      max_module_bytes=1024L,
      max_total_module_bytes=2048L,
      max_function_symbol_bytes=96,
    ),
  )
}

///|
fn fp8_bundle_receipt(
  candidate : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLowering,
  first : Bytes,
  second? : Bytes = fp8_bundle_bytes(),
) -> LunaDeterministicCompileReceipt {
  try! LunaDeterministicCompileReceipt::from_offline_builder(
    source_digest_sha256=candidate.source_digest().as_hex(),
    recipe_digest_sha256=candidate.recipe_digest().as_hex(),
    toolchain_digest_sha256="3".repeat(64),
    driver_identity_sha256="4".repeat(64),
    first_artifact_digest_sha256=kernel_bundle_sha256(first),
    second_artifact_digest_sha256=kernel_bundle_sha256(second),
  )
}
