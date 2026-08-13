///|
fn catalog_paged_plan() -> @plan.ModelPlan raise {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
  let layer = @plan.KvLayerId::new(0)
  @plan.ModelPlan::new_with_kv_execution(
    identity~,
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=64,
      vocabulary_size=128,
      hidden_size=16,
      intermediate_size=32,
      attention_head_count=4,
      kv_head_count=2,
      head_dimension=4,
    ),
    kv_cache_geometry=catalog_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[
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
    ],
    final_output=@plan.ActivationRef::new(2),
    kv_execution=PagedKeyValue,
  )
}

///|
test "v1 kernel catalog rejects paged semantics before shape matching" {
  let catalog = catalog_of([aot_entry(workspace_bytes=0L)])
  try catalog.resolve(catalog_paged_plan(), test_target()) catch {
    @catalog.UnsupportedExecutionContract => ()
    _ => fail("unexpected paged catalog admission failure")
  } noraise {
    _ => fail("expected paged execution contract rejection")
  }
}
