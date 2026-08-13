///|
fn paged_plan_fixture() -> @plan.ModelPlan raise {
  let layer = @plan.KvLayerId::new(0)
  let geometry = @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=2,
    head_dimension=4,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  @plan.ModelPlan::new_with_kv_execution(
    identity=contract_identity(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
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
fn paged_semantics_fixture(
  kind : @plan.OperationKind,
) -> @catalog.KernelExecutionSemantics raise {
  let layer = @plan.KvLayerId::new(0)
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
        runtime_inputs=[StepCounts],
        persistent_state=NoPersistentState,
        kv_layout=None,
      )
  }
}

///|
fn paged_catalog_fixture(
  rotary_workspace? : Int64 = 0L,
) -> @catalog.ResolvedKernelCatalog raise {
  let kinds : Array[@plan.OperationKind] = [
    Embedding,
    RotaryEmbedding,
    CausalAttention,
  ]
  let entries : Array[@catalog.CatalogEntry] = []
  for index, kind in kinds {
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
      _ => abort("unsupported fixture")
    }
    let workspace_bytes = if kind == RotaryEmbedding {
      rotary_workspace
    } else {
      0L
    }
    let workspace_alignment = if workspace_bytes > 0L { 16L } else { 1L }
    entries.push(
      @catalog.CatalogEntry::new_paged_v2(
        operation_kind=kind,
        operation_shape=shape,
        capability~,
        execution_semantics=paged_semantics_fixture(kind),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(contract_family(index + 11)),
        workspace_bytes~,
        workspace_alignment~,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v2(), entries~).resolve_paged(
    paged_plan_fixture(),
    paged_layout_fixture(),
    @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn paged_layout_fixture() -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    contract_identity(),
    paged_plan_fixture().kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn paged_profile_fixture() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
}

///|
fn paged_limits_fixture() -> @launch_contract.LaunchContractLimits raise {
  @launch_contract.LaunchContractLimits::new(
    max_profiles=8,
    max_contracts=32,
    max_operands_per_contract=16,
  )
}

///|
fn operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn paged_operands(operation : Int) -> Array[@launch_contract.AotOperand] raise {
  let layer = @plan.KvLayerId::new(0)
  match operation {
    0 =>
      [
        operand(RuntimeInput(StepCounts), 20L, 4L),
        operand(ExternalInput(TokenIds), 16L, 4L),
        operand(WeightInput(@plan.TensorRef::new(0)), 4096L, 16L),
        operand(ActivationOutput(@plan.ActivationRef::new(0)), 128L, 16L),
      ]
    1 =>
      [
        operand(RuntimeInput(StepCounts), 20L, 4L),
        operand(RuntimeInput(QueryPositions), 16L, 4L),
        operand(ActivationInput(@plan.ActivationRef::new(0)), 128L, 16L),
        operand(ActivationOutput(@plan.ActivationRef::new(1)), 128L, 16L),
      ]
    2 =>
      [
        operand(RuntimeInput(StepCounts), 20L, 4L),
        operand(RuntimeInput(QueryPositions), 16L, 4L),
        operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
        operand(RuntimeInput(SequenceLengths), 8L, 4L),
        operand(RuntimeInput(PageTableOffsets), 12L, 4L),
        operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
        operand(ActivationInput(@plan.ActivationRef::new(1)), 128L, 16L),
        operand(ActivationOutput(@plan.ActivationRef::new(2)), 128L, 16L),
        operand(PersistentKvComponent(layer, Key), 2048L, 256L),
        operand(PersistentKvComponent(layer, Value), 2048L, 256L),
      ]
    _ => abort("invalid fixture operation")
  }
}

///|
fn paged_inputs(
  attention_operands : ArrayView[@launch_contract.AotOperand],
) -> Array[@launch_contract.PagedKvAotLaunchContractInput] raise {
  let profile_id = @launch_contract.KernelProfileId::from_stable_int(1)
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation in [1, 2] {
    let operands = if operation == 2 {
      attention_operands
    } else {
      paged_operands(operation)
    }
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id~,
        operation_id=@plan.OperationId::new(operation),
        entry_point=entry_point(operation + 11, operation + 1),
        dimensions=dimensions(4),
        operands~,
      ),
    )
  }
  inputs
}

///|
test "paged v2 contract fixes canonical staged operand positions" {
  let attention = paged_operands(2)
  let set = @launch_contract.admit_paged_kv(
    paged_plan_fixture(),
    paged_catalog_fixture(),
    paged_layout_fixture(),
    [paged_profile_fixture()],
    paged_inputs(attention),
    paged_limits_fixture(),
  )
  assert_eq(set.catalog_version(), @catalog.CatalogVersion::v2())
  assert_eq(set.scope(), KvSubsequence)
  assert_eq(set.contracts().length(), 2)
  let contract = set.contracts()[1]
  assert_eq(contract.semantic_version(), @catalog.KernelSemanticVersion::v2())
  assert_eq(
    contract.operation_shape(),
    AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
  )
  assert_eq(contract.kv_layer(), Some(@plan.KvLayerId::new(0)))
  assert_eq(contract.kv_layout_version(), 1)
  assert_eq(contract.operands()[0].role(), RuntimeInput(StepCounts))
  assert_eq(contract.operands()[5].role(), RuntimeInput(PageTablePageIndices))
  assert_eq(
    contract.operands()[8].role(),
    PersistentKvComponent(@plan.KvLayerId::new(0), Key),
  )
  assert_eq(
    contract.operands()[9].role(),
    PersistentKvComponent(@plan.KvLayerId::new(0), Value),
  )
}

///|
test "paged v2 rejects missing, reordered, and foreign-layer operands" {
  let exact = paged_operands(2)
  let reordered = exact.copy()
  reordered[0] = exact[1]
  reordered[1] = exact[0]
  let foreign_layer = exact.copy()
  foreign_layer[8] = operand(
    PersistentKvComponent(@plan.KvLayerId::new(1), Key),
    2048L,
    256L,
  )
  let reversed_components = exact.copy()
  reversed_components[8] = exact[9]
  reversed_components[9] = exact[8]
  let cases : Array[
    (Array[@launch_contract.AotOperand], @launch_contract.OperandRule),
  ] = [
    (exact[:9].to_owned(), MissingSemanticRole),
    (reordered, PositionalRoleMismatch),
    (foreign_layer, PositionalRoleMismatch),
    (reversed_components, PositionalRoleMismatch),
  ]
  for case in cases {
    let (operands, expected) = case
    try
      @launch_contract.admit_paged_kv(
        paged_plan_fixture(),
        paged_catalog_fixture(),
        paged_layout_fixture(),
        [paged_profile_fixture()],
        paged_inputs(operands),
        paged_limits_fixture(),
      )
    catch {
      @launch_contract.InvalidOperand(_, actual) if actual == expected => ()
      _ => fail("unexpected paged operand failure")
    } noraise {
      _ => fail("expected paged operand rejection")
    }
  }
}

///|
test "paged KV set rejects extra non-KV contracts and requires both KV ops" {
  let exact = paged_inputs(paged_operands(2))
  let profile_id = @launch_contract.KernelProfileId::from_stable_int(1)
  let embedding = @launch_contract.PagedKvAotLaunchContractInput::new(
    profile_id~,
    operation_id=@plan.OperationId::new(0),
    entry_point=entry_point(11, 1),
    dimensions=dimensions(4),
    operands=paged_operands(0),
  )
  let extra = exact.copy()
  extra.push(embedding)
  try
    @launch_contract.admit_paged_kv(
      paged_plan_fixture(),
      paged_catalog_fixture(),
      paged_layout_fixture(),
      [paged_profile_fixture()],
      extra,
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.ContractForNonPagedKvOperation(_, id) =>
      assert_eq(id.as_int(), 0)
    _ => fail("unexpected non-KV contract error")
  } noraise {
    _ => fail("expected non-KV contract rejection")
  }
  try
    @launch_contract.admit_paged_kv(
      paged_plan_fixture(),
      paged_catalog_fixture(),
      paged_layout_fixture(),
      [paged_profile_fixture()],
      exact[:1],
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.MissingContract(_, id) => assert_eq(id.as_int(), 2)
    _ => fail("unexpected missing paged-KV contract error")
  } noraise {
    _ => fail("expected missing paged attention contract")
  }
}

///|
test "paged v2 rejects wrong layout, mode, and catalog version" {
  let foreign_geometry = @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=1,
    head_dimension=4,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let foreign_layout = @device_layout.plan(
    contract_identity(),
    foreign_geometry,
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
  try
    @launch_contract.admit_paged_kv(
      paged_plan_fixture(),
      paged_catalog_fixture(),
      foreign_layout,
      [paged_profile_fixture()],
      paged_inputs(paged_operands(2)),
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.DeviceKvLayoutMismatch => ()
    _ => fail("unexpected foreign-layout failure")
  } noraise {
    _ => fail("expected foreign layout rejection")
  }
  try
    @launch_contract.admit_paged_kv(
      contract_plan(),
      contract_catalog(),
      paged_layout_fixture(),
      [paged_profile_fixture()],
      [],
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.UnsupportedExecutionContract => ()
    _ => fail("unexpected wrong-mode failure")
  } noraise {
    _ => fail("expected stateless mode rejection")
  }
  try
    @launch_contract.admit_paged_kv(
      paged_plan_fixture(),
      contract_catalog(),
      paged_layout_fixture(),
      [paged_profile_fixture()],
      [],
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.CatalogVersionMismatch => ()
    _ => fail("unexpected catalog-version failure")
  } noraise {
    _ => fail("expected v1 catalog rejection")
  }
}
