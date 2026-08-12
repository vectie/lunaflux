///|
fn contract_identity() -> @spec.ModelIdentity {
  @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    try! @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn contract_plan() -> @plan.ModelPlan {
  let embedding = try! @plan.PlanOperation::new(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let norm = try! @plan.PlanOperation::new(
    id=@plan.OperationId::new(1),
    kind=RmsNorm,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=NormalizationShape(width=16, epsilon=0.000001),
    required_capability=@plan.KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
  )
  try! @plan.ModelPlan::new(
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
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[embedding, norm],
    final_output=@plan.ActivationRef::new(1),
  )
}

///|
fn contract_digest() -> @catalog.AotArtifactDigest {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  digest
}

///|
fn contract_family(stable_id : Int) -> @catalog.AotKernelFamily {
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(stable_id)
  @catalog.AotKernelFamily::new(module_digest=contract_digest(), family_id~)
}

///|
fn contract_catalog() -> @catalog.ResolvedKernelCatalog {
  let embedding = try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(contract_family(1)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  let norm = try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=RmsNorm,
    operation_shape=NormalizationShape(width=16, epsilon=0.000001),
    capability=@plan.KernelCapabilityId::rms_norm(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(contract_family(2)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  try! @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries=[
    norm, embedding,
  ]).resolve(
    contract_plan(),
    @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn profile(stable_id : Int, rows : Int) -> @launch_contract.KernelProfile {
  let id = try! @launch_contract.KernelProfileId::from_stable_int(stable_id)
  try! @launch_contract.KernelProfile::full_prefill(
    id~,
    batch_rows=1,
    sequence_tokens=rows,
    token_rows=rows.to_int64(),
  )
}

///|
fn dimensions(rows : Int) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=rows,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn entry_point(family : Int, entry : Int) -> @catalog.AotKernelEntryPoint {
  let entry_point_id = try! @catalog.AotKernelEntryPointId::from_stable_int(
    entry,
  )
  @catalog.AotKernelEntryPoint::new(
    family=contract_family(family),
    entry_point_id~,
  )
}

///|
fn embedding_operands(rows : Int) -> Array[@launch_contract.AotOperand] {
  [
    try! @launch_contract.AotOperand::new(
      role=ExternalInput(TokenIds),
      byte_count=rows.to_int64() * 4L,
      alignment=4L,
    ),
    try! @launch_contract.AotOperand::new(
      role=WeightInput(@plan.TensorRef::new(0)),
      byte_count=4096L,
      alignment=16L,
    ),
    try! @launch_contract.AotOperand::new(
      role=ActivationOutput(@plan.ActivationRef::new(0)),
      byte_count=rows.to_int64() * 32L,
      alignment=16L,
    ),
  ]
}

///|
fn norm_operands(rows : Int) -> Array[@launch_contract.AotOperand] {
  [
    try! @launch_contract.AotOperand::new(
      role=ActivationInput(@plan.ActivationRef::new(0)),
      byte_count=rows.to_int64() * 32L,
      alignment=16L,
    ),
    try! @launch_contract.AotOperand::new(
      role=WeightInput(@plan.TensorRef::new(1)),
      byte_count=32L,
      alignment=16L,
    ),
    try! @launch_contract.AotOperand::new(
      role=ActivationOutput(@plan.ActivationRef::new(1)),
      byte_count=rows.to_int64() * 32L,
      alignment=16L,
    ),
  ]
}

///|
fn input(
  profile_id : Int,
  operation_id : Int,
  family : Int,
  entry : Int,
  rows : Int,
  operands : ArrayView[@launch_contract.AotOperand],
) -> @launch_contract.AotLaunchContractInput {
  let resolved_profile_id = try! @launch_contract.KernelProfileId::from_stable_int(
    profile_id,
  )
  let resolved_operation_id = try! @plan.OperationId::new(operation_id)
  @launch_contract.AotLaunchContractInput::new(
    profile_id=resolved_profile_id,
    operation_id=resolved_operation_id,
    entry_point=entry_point(family, entry),
    dimensions=dimensions(rows),
    operands~,
  )
}

///|
fn contract_limits() -> @launch_contract.LaunchContractLimits {
  try! @launch_contract.LaunchContractLimits::new(
    max_profiles=8,
    max_contracts=32,
    max_operands_per_contract=8,
  )
}

///|
test "exact profiles select distinct entry points in deterministic order" {
  let small = profile(1, 2)
  let large = profile(2, 8)
  let inputs = [
    input(2, 1, 2, 22, 8, norm_operands(8)),
    input(1, 0, 1, 11, 2, embedding_operands(2)),
    input(2, 0, 1, 12, 8, embedding_operands(8)),
    input(1, 1, 2, 21, 2, norm_operands(2)),
  ]
  let profiles = [small, large]
  let set = @launch_contract.admit(
    contract_plan(),
    contract_catalog(),
    profiles,
    inputs,
    contract_limits(),
  )
  inputs[0] = inputs[1]
  profiles[0] = large
  assert_eq(set.contracts().length(), 4)
  assert_eq(set.profiles()[0], small)
  assert_eq(set.contracts()[0].profile(), small)
  assert_eq(set.contracts()[0].operation_id().as_int(), 0)
  assert_eq(set.contracts()[0].entry_point().entry_point_id().as_int(), 11)
  assert_eq(set.contracts()[2].entry_point().entry_point_id().as_int(), 12)
  let norm_id = try! @plan.OperationId::new(1)
  guard set.contract(large.id(), norm_id) is Some(found) else {
    fail("expected exact integer lookup")
  }
  assert_eq(found.dimensions().grid_x(), 8)
}

///|
test "operand tables are defensively owned and semantically exact" {
  let operands = embedding_operands(2)
  let embedding = input(1, 0, 1, 11, 2, operands)
  operands[0] = operands[1]
  let set = @launch_contract.admit(
    contract_plan(),
    contract_catalog(),
    [profile(1, 2)],
    [embedding, input(1, 1, 2, 21, 2, norm_operands(2))],
    contract_limits(),
  )
  assert_eq(set.contracts()[0].operands()[0].role(), ExternalInput(TokenIds))
  let missing = input(1, 0, 1, 11, 2, embedding_operands(2)[:2])
  try
    @launch_contract.admit(
      contract_plan(),
      contract_catalog(),
      [profile(1, 2)],
      [missing, input(1, 1, 2, 21, 2, norm_operands(2))],
      contract_limits(),
    )
  catch {
    InvalidOperand(_, MissingSemanticRole) => ()
    _ => fail("unexpected launch-contract error")
  } noraise {
    _ => fail("expected missing semantic operand")
  }
}

///|
test "profile and contract ambiguity fail before execution" {
  let exact = input(1, 0, 1, 11, 2, embedding_operands(2))
  let norm = input(1, 1, 2, 21, 2, norm_operands(2))
  try
    @launch_contract.admit(
      contract_plan(),
      contract_catalog(),
      [profile(1, 2)],
      [exact, exact, norm],
      contract_limits(),
    )
  catch {
    DuplicateContract(profile_id, operation_id) => {
      assert_eq(profile_id.as_int(), 1)
      assert_eq(operation_id.as_int(), 0)
    }
    _ => fail("unexpected launch-contract error")
  } noraise {
    _ => fail("expected duplicate contract")
  }
  try
    @launch_contract.admit(
      contract_plan(),
      contract_catalog(),
      [profile(1, 2), profile(1, 2)],
      [exact, norm],
      contract_limits(),
    )
  catch {
    DuplicateProfile(profile_id) => assert_eq(profile_id.as_int(), 1)
    _ => fail("unexpected launch-contract error")
  } noraise {
    _ => fail("expected duplicate profile")
  }
  let wrong_family = input(1, 0, 2, 11, 2, embedding_operands(2))
  try
    @launch_contract.admit(
      contract_plan(),
      contract_catalog(),
      [profile(1, 2)],
      [wrong_family, norm],
      contract_limits(),
    )
  catch {
    KernelFamilyMismatch(profile_id, operation_id) => {
      assert_eq(profile_id.as_int(), 1)
      assert_eq(operation_id.as_int(), 0)
    }
    _ => fail("unexpected launch-contract error")
  } noraise {
    _ => fail("expected exact AOT family validation")
  }
  try
    @launch_contract.admit(
      contract_plan(),
      contract_catalog(),
      [profile(1, 2)],
      [exact],
      contract_limits(),
    )
  catch {
    MissingContract(profile_id, operation_id) => {
      assert_eq(profile_id.as_int(), 1)
      assert_eq(operation_id.as_int(), 1)
    }
    _ => fail("unexpected launch-contract error")
  } noraise {
    _ => fail("expected missing contract")
  }
  let invalid_id = try! @launch_contract.KernelProfileId::from_stable_int(3)
  try
    @launch_contract.KernelProfile::full_prefill(
      id=invalid_id,
      batch_rows=2,
      sequence_tokens=4,
      token_rows=7L,
    )
  catch {
    InvalidProfile(TokenRowMismatch) => ()
    _ => fail("unexpected launch-contract error")
  } noraise {
    _ => fail("expected exact token-row validation")
  }
}
