///|
fn readonly_fixture_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256("0".repeat(64)),
    @spec.PlanDigest::from_sha256("1".repeat(64)),
  )
}

///|
fn readonly_fixture_operation() -> @plan.PlanOperation raise {
  let layer = @plan.KvLayerId::new(0)
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
  )
}

///|
fn readonly_fixture_layout(
  tokens_per_page? : Int = 2,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    readonly_fixture_identity(),
    @plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page~,
      total_page_count=160,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn readonly_fixture_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=4,
    max_query_tokens=260,
    max_page_table_entries=132,
  )
}

///|
fn readonly_fixture_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn readonly_fixture_operands() -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  [
    readonly_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    readonly_fixture_operand(RuntimeInput(QueryPositions), 1040L, 4L),
    readonly_fixture_operand(RuntimeInput(QueryRowOffsets), 20L, 4L),
    readonly_fixture_operand(RuntimeInput(SequenceLengths), 16L, 4L),
    readonly_fixture_operand(RuntimeInput(PageTableOffsets), 20L, 4L),
    readonly_fixture_operand(RuntimeInput(PageTablePageIndices), 528L, 4L),
    readonly_fixture_operand(
      ActivationInput(@plan.ActivationRef::new(1)),
      4160L,
      16L,
    ),
    readonly_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(2)),
      2080L,
      16L,
    ),
    readonly_fixture_operand(PersistentKvComponent(layer, Key), 40960L, 256L),
    readonly_fixture_operand(PersistentKvComponent(layer, Value), 40960L, 256L),
  ]
}

///|
fn readonly_fixture_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn readonly_fixture_compiler(
  toolchain_sha256 : String,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=toolchain_sha256,
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn readonly_fixture_reference(
  toolchain_sha256 : String,
) -> @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering raise {
  @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
    readonly_fixture_operation(),
    readonly_fixture_profile(),
    readonly_fixture_layout(),
    @catalog.AotKernelEntryPointId::from_stable_int(3),
    readonly_fixture_target(),
    readonly_fixture_compiler(toolchain_sha256),
    readonly_fixture_operands(),
    @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=8192,
    ),
  )
}

///|
fn readonly_fixture_candidate(
  toolchain_sha256 : String,
  entry? : Int = 103,
) -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate raise {
  @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
    readonly_fixture_reference(toolchain_sha256),
    @catalog.AotKernelEntryPointId::from_stable_int(entry),
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
      maximum_source_bytes=65536,
      maximum_recipe_bytes=8192,
      maximum_canonical_bytes=8192,
    ),
  )
}
