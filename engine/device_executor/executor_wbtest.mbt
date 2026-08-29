///|
fn executor_test_metadata(
  digest? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(digest),
  )
}

///|
fn executor_test_semantic_plan(
  metadata : @spec.LlamaModelMetadata,
) -> @plan.ModelPlan raise {
  let embedding = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let head = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(1),
    kind=LanguageModelHead,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(0)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=ProjectionShape(input_width=4, output_width=8),
    required_capability=@plan.KernelCapabilityId::dense_projection(),
    workspace_bytes=0L,
  )
  @plan.ModelPlan::new_stateless(
    content_digest=metadata.content_digest(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=4,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    kv_cache_geometry=executor_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[embedding, head],
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema([
      embedding, head,
    ]),
  )
}

///|
fn executor_test_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn executor_test_catalog(
  semantic : @plan.ModelPlan,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let family = @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(1),
  )
  let entries = [
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=Embedding,
      operation_shape=semantic.operations()[0].shape(),
      capability=@plan.KernelCapabilityId::embedding_lookup(),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(family),
      workspace_bytes=0L,
      workspace_alignment=16L,
    ),
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=LanguageModelHead,
      operation_shape=semantic.operations()[1].shape(),
      capability=@plan.KernelCapabilityId::dense_projection(),
      compute_major=8,
      compute_minor=0,
      implementation=VendorBf16GemmImplicitHeuristicV1,
      workspace_bytes=0L,
      workspace_alignment=16L,
    ),
  ]
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries~).resolve(
    semantic,
    executor_test_target(),
  )
}

///|
priv struct ExecutorTestFixture {
  static_plan : @device_plan.StaticDevicePlan
  memory_plan : @device_memory.DeviceMemoryPlan
  profile : @device_profile.DeviceExecutionProfile
  profile_id : @launch_contract.KernelProfileId
  contracts : @launch_contract.LaunchContractSet
  artifacts : @artifact.KernelArtifactBundle
  layout : @device_materialize.DeviceWeightLayout
}

///|
fn executor_test_fixture(
  output_bytes? : Int64 = 32L,
) -> ExecutorTestFixture raise {
  let metadata = executor_test_metadata()
  let semantic = executor_test_semantic_plan(metadata)
  let layout = @device_materialize.plan_numeric_layout(
    semantic,
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
  )
  let resolved = executor_test_catalog(semantic, digest)
  let static_plan = @device_plan.build(semantic, layout, resolved)
  let memory_plan = @device_memory.plan(
    static_plan,
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=4096L,
    ),
  )
  let profile = @device_profile.build(
    static_plan,
    memory_plan,
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=1,
      sequence_tokens=2,
    ),
  )
  let profile_id = @launch_contract.KernelProfileId::from_stable_int(1)
  let kernel_profile = @launch_contract.KernelProfile::full_prefill(
    id=profile_id,
    batch_rows=1,
    sequence_tokens=2,
    token_rows=2L,
  )
  let family = match resolved.bindings()[0].implementation() {
    ContentAddressedAot(value) => value
    _ => fail("expected AOT embedding")
  }
  let entry_point = @catalog.AotKernelEntryPoint::new(
    family~,
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(1),
  )
  let contract_input = @launch_contract.AotLaunchContractInput::new(
    profile_id~,
    operation_id=@plan.OperationId::new(0),
    entry_point~,
    dimensions=@launch_contract.AotLaunchDimensions::new(
      grid_x=1,
      grid_y=1,
      grid_z=1,
      block_x=32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    operands=[
      @launch_contract.AotOperand::new(
        role=ExternalInput(TokenIds),
        byte_count=8L,
        alignment=4L,
      ),
      @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(0)),
        byte_count=64L,
        alignment=64L,
      ),
      @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(0)),
        byte_count=output_bytes,
        alignment=64L,
      ),
    ],
  )
  let contracts = @launch_contract.admit(
    semantic,
    resolved,
    [kernel_profile],
    [contract_input],
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=4,
      max_operands_per_contract=8,
    ),
  )
  let artifacts = @artifact.admit(
    contracts,
    [
      @artifact.KernelModuleInput::new(
        digest~,
        module_bytes=FixedArray::from_array([b'a']),
      ),
    ],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point~,
        function_symbol="embed_exact",
      ),
    ],
    @artifact.KernelArtifactLimits::new(
      max_modules=2,
      max_entry_points=4,
      max_module_bytes=1024L,
      max_total_module_bytes=2048L,
      max_function_symbol_bytes=64,
    ),
  )
  {
    static_plan,
    memory_plan,
    profile,
    profile_id,
    contracts,
    artifacts,
    layout,
  }
}

///|
test "blueprint preserves exact operand order, semantic order, and zero workspace omission" {
  let fixture = executor_test_fixture(output_bytes=16L)
  let blueprint = build_blueprint_from_evidence(
    fixture.static_plan,
    fixture.memory_plan,
    fixture.profile,
    fixture.profile_id,
    fixture.layout,
    fixture.layout.arena_bytes(),
    fixture.contracts,
    fixture.artifacts,
    context_target_matches=true,
  )
  assert_eq(blueprint.steps.length(), 2)
  guard blueprint.steps is [Aot(aot), Vendor(vendor)] else {
    fail("unexpected semantic step order")
  }
  assert_eq(aot.operands.length(), 3)
  assert_eq(aot.operands[0].allocation_space, TokenInput)
  assert_eq(aot.operands[0].byte_count, 8L)
  assert_eq(aot.operands[1].allocation_space, WeightArena)
  assert_eq(aot.operands[1].offset_bytes, 0L)
  assert_eq(aot.operands[1].byte_count, 64L)
  assert_eq(aot.operands[2].allocation_space, ActivationWorkspaceArena)
  assert_eq(aot.operands[2].byte_count, 16L)
  assert_eq(vendor.rows, 2L)
  assert_eq(vendor.inner, 4L)
  assert_eq(vendor.columns, 8L)
  assert_eq(vendor.workspace_bytes, 0L)
}

///|
test "validation failure occurs before the resource continuation" {
  let fixture = executor_test_fixture(output_bytes=16L)
  let opened = Ref(0)
  try {
    ignore(
      build_blueprint_from_evidence(
        fixture.static_plan,
        fixture.memory_plan,
        fixture.profile,
        fixture.profile_id,
        fixture.layout,
        fixture.layout.arena_bytes(),
        fixture.contracts,
        fixture.artifacts,
        context_target_matches=false,
      ),
    )
    opened.val += 1
  } catch {
    InvalidPlan(ContextCapability) => assert_eq(opened.val, 0)
    error => fail("unexpected failure: \{Repr(error)}")
  } noraise {
    _ => fail("target mismatch unexpectedly opened resources")
  }
}

///|
test "exact physical operand bytes are mandatory" {
  let fixture = executor_test_fixture(output_bytes=32L)
  try
    build_blueprint_from_evidence(
      fixture.static_plan,
      fixture.memory_plan,
      fixture.profile,
      fixture.profile_id,
      fixture.layout,
      fixture.layout.arena_bytes(),
      fixture.contracts,
      fixture.artifacts,
      context_target_matches=true,
    )
  catch {
    InvalidOperation(operation, OperandSize) => assert_eq(operation.as_int(), 0)
    error => fail("unexpected failure: \{Repr(error)}")
  } noraise {
    _ => fail("mismatched operand bytes were admitted")
  }
}

///|
test "token validation rejects negative and out-of-vocabulary values" {
  validate_token_ids(b"\x00\x00\x00\x00\x07\x00\x00\x00", 8)
  try validate_token_ids(b"\xff\xff\xff\xff", 8) catch {
    InvalidTokenId(index=0) => ()
    error => fail("unexpected failure: \{Repr(error)}")
  } noraise {
    _ => fail("negative token was admitted")
  }
  try validate_token_ids(b"\x08\x00\x00\x00", 8) catch {
    InvalidTokenId(index=0) => ()
    error => fail("unexpected failure: \{Repr(error)}")
  } noraise {
    _ => fail("out-of-vocabulary token was admitted")
  }
}

///|
priv struct FakeSibling {
  id : Int
  mut open : Bool
}

///|
test "production sibling closer is reverse, retry-safe, and idempotent" {
  let siblings : Array[FakeSibling] = [
    { id: 1, open: true },
    { id: 2, open: true },
    { id: 3, open: true },
  ]
  let order : Array[Int] = []
  let fail_once = Ref(true)
  let first : Ref[DeviceExecutorError?] = Ref(None)
  close_siblings_reverse(
    siblings,
    sibling => sibling.open,
    sibling => {
      order.push(sibling.id)
      if sibling.id == 2 && fail_once.val {
        fail_once.val = false
        raise CleanupFailed(Function, Device(DriverFailure(FunctionClose)))
      }
    },
    sibling => sibling.open = false,
    first,
  )
  assert_eq(order, [3, 2, 1])
  assert_true(first.val is Some(CleanupFailed(Function, _)))
  assert_true(siblings[1].open)
  first.val = None
  close_siblings_reverse(
    siblings,
    sibling => sibling.open,
    sibling => order.push(sibling.id),
    sibling => sibling.open = false,
    first,
  )
  assert_eq(order, [3, 2, 1, 2])
  assert_true(first.val is None)
  close_siblings_reverse(
    siblings,
    sibling => sibling.open,
    sibling => order.push(sibling.id),
    sibling => sibling.open = false,
    first,
  )
  assert_eq(order, [3, 2, 1, 2])
}

///|
