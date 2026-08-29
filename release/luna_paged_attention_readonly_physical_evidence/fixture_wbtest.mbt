///|
fn readonly_evidence_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn readonly_evidence_candidate() -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate raise {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256("0".repeat(64)),
    @spec.PlanDigest::from_sha256("1".repeat(64)),
  )
  let layer = @plan.KvLayerId::new(0)
  let operation = @plan.PlanOperation::new_with_execution(
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
  let layout = @device_layout.plan(
    identity,
    @plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=2,
      total_page_count=160,
      max_arena_bytes=1048576L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=4,
    max_query_tokens=260,
    max_page_table_entries=132,
  )
  let operands = [
    readonly_evidence_operand(RuntimeInput(StepCounts), 20L, 4L),
    readonly_evidence_operand(RuntimeInput(QueryPositions), 1040L, 4L),
    readonly_evidence_operand(RuntimeInput(QueryRowOffsets), 20L, 4L),
    readonly_evidence_operand(RuntimeInput(SequenceLengths), 16L, 4L),
    readonly_evidence_operand(RuntimeInput(PageTableOffsets), 20L, 4L),
    readonly_evidence_operand(RuntimeInput(PageTablePageIndices), 528L, 4L),
    readonly_evidence_operand(
      ActivationInput(@plan.ActivationRef::new(1)),
      4160L,
      16L,
    ),
    readonly_evidence_operand(
      ActivationOutput(@plan.ActivationRef::new(2)),
      2080L,
      16L,
    ),
    readonly_evidence_operand(PersistentKvComponent(layer, Key), 40960L, 256L),
    readonly_evidence_operand(PersistentKvComponent(layer, Value), 40960L, 256L),
  ]
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=readonly_physical_digest(
        @utf8.encode("driver identity"),
      ).as_hex(),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let fallback = @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
    operation,
    profile,
    layout,
    @catalog.AotKernelEntryPointId::from_stable_int(3),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    ),
    compiler,
    operands,
    @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=8192,
    ),
  )
  @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
    fallback,
    @catalog.AotKernelEntryPointId::from_stable_int(103),
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
      maximum_source_bytes=65536,
      maximum_recipe_bytes=8192,
      maximum_canonical_bytes=8192,
    ),
  )
}

///|
fn readonly_evidence_compiled(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
) -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding raise {
  let module_bytes = b"ELF-CUBIN-readonly-physical-test"
  let module_digest = readonly_physical_digest(module_bytes).as_hex()
  let fallback = candidate.reference_fallback()
  let receipt = @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-aot-evidence.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest().as_hex() +
    "\nsource_sha256=" +
    candidate.source_digest().as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\nfallback_source_sha256=" +
    fallback.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    fallback.recipe_digest().as_hex() +
    "\ntoolchain_sha256=" +
    fallback.compiler_policy().identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    fallback.compiler_policy().identity().toolchain_digest_sha256() +
    "\ntarget=sm_120\nfunction_symbol=" +
    candidate.function_symbol() +
    "\nraw_pointer_abi=" +
    candidate.kernel_abi().canonical_name() +
    "\ninput_row_width=8\ninput_activation_ref=1\ndispatch_canary_per_token=65624\n" +
    "dispatch_canary_cell_count=260\nartifact_sha256=" +
    module_digest +
    "\nfirst_build_sha256=" +
    module_digest +
    "\nsecond_build_sha256=" +
    module_digest +
    "\ndeterministic=1\nkv_cache_mutation=none\n",
  )
  @luna_cuda_paged_attention_readonly_aot.bind_paged_attention_readonly_compiled_candidate(
    candidate,
    module_bytes,
    module_bytes,
    receipt,
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyDigest::from_sha256(
      readonly_physical_digest(receipt).as_hex(),
    ),
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledLimits::new(
      maximum_module_bytes=4096L,
      maximum_receipt_bytes=8192,
      maximum_binding_bytes=8192,
    ),
  )
}
