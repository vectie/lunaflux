///|
fn paged_v2_semantics(
  kind : @plan.OperationKind,
  layer : @plan.KvLayerId,
) -> @catalog.KernelExecutionSemantics raise {
  match kind {
    RotaryEmbedding =>
      @catalog.KernelExecutionSemantics::paged_v2(
        kv_layer=Some(layer),
        runtime_inputs=[StepCounts, QueryPositions],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
    CausalAttention =>
      @catalog.KernelExecutionSemantics::paged_v2(
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
    _ =>
      @catalog.KernelExecutionSemantics::paged_v2(
        kv_layer=None,
        runtime_inputs=[],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
  }
}

///|
fn paged_v2_entry(
  kind : @plan.OperationKind,
  layer : @plan.KvLayerId,
  family_id : Int,
) -> @catalog.CatalogEntry raise {
  let (shape, capability) = match kind {
    Embedding =>
      (
        @plan.EmbeddingShape(vocabulary_size=128, width=16),
        @plan.KernelCapabilityId::embedding_lookup(),
      )
    RotaryEmbedding =>
      (
        RotaryShape(
          query_heads=4,
          kv_heads=2,
          head_dimension=4,
          rope_theta=10000.0,
        ),
        @plan.KernelCapabilityId::positioned_rotary_embedding(),
      )
    CausalAttention =>
      (
        AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
        @plan.KernelCapabilityId::paged_causal_attention(),
      )
    _ => abort("unsupported fixture kind")
  }
  @catalog.CatalogEntry::new_paged_v2(
    operation_kind=kind,
    operation_shape=shape,
    capability~,
    execution_semantics=paged_v2_semantics(kind, layer),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(
      @catalog.AotKernelFamily::new(
        module_digest=@catalog.AotArtifactDigest::from_sha256(
          "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        ),
        family_id=@catalog.AotKernelFamilyId::from_stable_int(family_id),
      ),
    ),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
}

///|
fn paged_v2_catalog(layer : @plan.KvLayerId) -> @catalog.KernelCatalog raise {
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v2(), entries=[
    paged_v2_entry(Embedding, layer, 1),
    paged_v2_entry(RotaryEmbedding, layer, 2),
    paged_v2_entry(CausalAttention, layer, 3),
  ])
}

///|
fn paged_v2_layout(
  tokens_per_page? : Int = 4,
) -> @device_layout.DeviceKvLayout raise {
  let plan = catalog_paged_plan()
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page~,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
}

///|
test "v2 resolves exact paged semantics and preserves binding ABI" {
  let layer = @plan.KvLayerId::new(0)
  let resolved = paged_v2_catalog(layer).resolve_paged(
    catalog_paged_plan(),
    paged_v2_layout(),
    test_target(),
  )
  assert_eq(resolved.catalog_version(), @catalog.CatalogVersion::v2())
  let rotary = resolved.bindings()[1]
  assert_eq(rotary.semantic_version(), @catalog.KernelSemanticVersion::v2())
  guard rotary.execution_semantics() is Some(semantics) else {
    fail("expected v2 execution semantics")
  }
  assert_eq(semantics.kv_layer(), Some(layer))
  assert_eq(semantics.runtime_inputs(), [StepCounts, QueryPositions])
  let attention = resolved.bindings()[2]
  guard attention.execution_semantics() is Some(attention_semantics) else {
    fail("expected attention semantics")
  }
  assert_eq(attention_semantics.persistent_state(), KvCacheReadWrite(layer))
  guard attention_semantics.kv_layout() is Some(layout) else {
    fail("expected paged KV layout contract")
  }
  assert_eq(layout.layout(), LayerMajorSplitKeyValueV1)
  assert_eq(layout.stable_version(), 1)
  assert_eq(layout.tokens_per_page(), 4)
}

///|
test "v2 entries reject missing and reordered live roles" {
  let layer = @plan.KvLayerId::new(0)
  let invalid_inputs : Array[Array[@plan.OperationRuntimeInput]] = [
    [StepCounts, QueryPositions, QueryRowOffsets, SequenceLengths],
    [
      QueryPositions,
      StepCounts,
      QueryRowOffsets,
      SequenceLengths,
      PageTableOffsets,
      PageTablePageIndices,
    ],
  ]
  for runtime_inputs in invalid_inputs {
    let semantics = @catalog.KernelExecutionSemantics::paged_v2(
      kv_layer=Some(layer),
      runtime_inputs~,
      persistent_state=KvCacheReadWrite(layer),
      kv_layout=Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page=4,
        ),
      ),
    )
    try
      @catalog.CatalogEntry::new_paged_v2(
        operation_kind=CausalAttention,
        operation_shape=AttentionShape(
          query_heads=4,
          kv_heads=2,
          head_dimension=4,
        ),
        capability=@plan.KernelCapabilityId::paged_causal_attention(),
        execution_semantics=semantics,
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(
          @catalog.AotKernelFamily::new(
            module_digest=@catalog.AotArtifactDigest::from_sha256(
              "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            ),
            family_id=@catalog.AotKernelFamilyId::from_stable_int(9),
          ),
        ),
        workspace_bytes=0L,
        workspace_alignment=1L,
      )
    catch {
      InvalidCatalog(InvalidExecutionSemantics) => ()
      _ => fail("unexpected v2 semantic validation error")
    } noraise {
      _ => fail("expected invalid v2 runtime order")
    }
  }
}

///|
test "v2 family reuse normalizes layer identity but retains physical ABI" {
  let layer_zero = @plan.KvLayerId::new(0)
  let layer_one = @plan.KvLayerId::new(1)
  let first = paged_v2_entry(CausalAttention, layer_zero, 41)
  let second = paged_v2_entry(CausalAttention, layer_one, 41)
  let reused = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v2(),
    entries=[first, second],
  )
  assert_eq(reused.entries().length(), 2)

  let foreign_layout = @catalog.KernelExecutionSemantics::paged_v2(
    kv_layer=Some(layer_one),
    runtime_inputs=[
      StepCounts,
      QueryPositions,
      QueryRowOffsets,
      SequenceLengths,
      PageTableOffsets,
      PageTablePageIndices,
    ],
    persistent_state=KvCacheReadWrite(layer_one),
    kv_layout=Some(
      @catalog.PagedKvLayoutContract::new(
        layout=LayerMajorSplitKeyValueV1,
        stable_version=1,
        tokens_per_page=8,
      ),
    ),
  )
  let mismatch = @catalog.CatalogEntry::new_paged_v2(
    operation_kind=CausalAttention,
    operation_shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
    capability=@plan.KernelCapabilityId::paged_causal_attention(),
    execution_semantics=foreign_layout,
    compute_major=8,
    compute_minor=0,
    implementation=second.implementation(),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  try
    @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v2(), entries=[
      first, mismatch,
    ])
  catch {
    InvalidCatalog(AotFamilySemanticMismatch) => ()
    _ => fail("unexpected family physical-ABI error")
  } noraise {
    _ => fail("expected physical ABI mismatch")
  }
}

///|
test "v2 resolution rejects tokens-per-page specialization mismatch" {
  let layer = @plan.KvLayerId::new(0)
  try
    paged_v2_catalog(layer).resolve_paged(
      catalog_paged_plan(),
      paged_v2_layout(tokens_per_page=2),
      test_target(),
    )
  catch {
    @catalog.MissingKernel(id) => assert_eq(id.as_int(), 2)
    _ => fail("unexpected tokens-per-page error")
  } noraise {
    _ => fail("expected tokens-per-page rejection")
  }
}

///|
test "catalog versions, modes, and layer identities do not cross" {
  let layer = @plan.KvLayerId::new(0)
  try
    @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v2(), entries=[
      aot_entry(),
    ])
  catch {
    InvalidCatalog(CatalogSemanticVersionMismatch) => ()
    _ => fail("unexpected mixed-version error")
  } noraise {
    _ => fail("expected mixed catalog rejection")
  }
  try paged_v2_catalog(layer).resolve(test_plan(0L), test_target()) catch {
    @catalog.UnsupportedExecutionContract => ()
    _ => fail("unexpected execution-mode error")
  } noraise {
    _ => fail("expected v2 stateless-plan rejection")
  }
  let foreign = @plan.KvLayerId::new(1)
  try
    paged_v2_catalog(foreign).resolve_paged(
      catalog_paged_plan(),
      paged_v2_layout(),
      test_target(),
    )
  catch {
    @catalog.MissingKernel(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected foreign-layer error")
  } noraise {
    _ => fail("expected foreign semantic layer rejection")
  }
}

///|
test "v1 matching preserves stateless full-context layer semantics" {
  let layer = @plan.KvLayerId::new(0)
  let plan = @plan.ModelPlan::new_with_kv_execution(
    identity=test_identity(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
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
      @plan.PlanOperation::new_stateless(
        id=@plan.OperationId::new(0),
        kind=Embedding,
        inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
        outputs=[@plan.ActivationRef::new(0)],
        shape=EmbeddingShape(vocabulary_size=128, width=16),
        required_capability=@plan.KernelCapabilityId::embedding_lookup(),
        workspace_bytes=0L,
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
        required_capability=@plan.KernelCapabilityId::rotary_embedding(),
        workspace_bytes=0L,
        kv_layer=layer,
        runtime_inputs=[],
        persistent_state=NoPersistentState,
      ),
      @plan.PlanOperation::new_with_execution(
        id=@plan.OperationId::new(2),
        kind=CausalAttention,
        inputs=[Activation(@plan.ActivationRef::new(1))],
        outputs=[@plan.ActivationRef::new(2)],
        shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
        required_capability=@plan.KernelCapabilityId::causal_attention(),
        workspace_bytes=0L,
        kv_layer=layer,
        runtime_inputs=[],
        persistent_state=NoPersistentState,
      ),
    ],
    final_output=@plan.ActivationRef::new(2),
    kv_execution=StatelessFullContext,
  )
  let entries : Array[@catalog.CatalogEntry] = []
  for operation in plan.operations() {
    entries.push(
      @catalog.CatalogEntry::new(
        semantic_version=@catalog.KernelSemanticVersion::v1(),
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(
          @catalog.AotKernelFamily::new(
            module_digest=@catalog.AotArtifactDigest::from_sha256(
              "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            ),
            family_id=@catalog.AotKernelFamilyId::from_stable_int(
              operation.id().as_int() + 1,
            ),
          ),
        ),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  let resolved = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries~,
  ).resolve(plan, test_target())
  assert_eq(resolved.bindings().length(), 3)
}
