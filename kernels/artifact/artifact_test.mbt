///|
fn artifact_identity() -> @spec.ModelIdentity {
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
fn artifact_plan() -> @plan.ModelPlan {
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
    identity=artifact_identity(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[embedding, norm],
    final_output=@plan.ActivationRef::new(1),
  )
}

///|
fn digest_for(bytes : FixedArray[Byte]) -> @catalog.AotArtifactDigest {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  digest
}

///|
fn family(
  digest : @catalog.AotArtifactDigest,
  stable_id : Int,
) -> @catalog.AotKernelFamily {
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(stable_id)
  @catalog.AotKernelFamily::new(module_digest=digest, family_id~)
}

///|
fn resolved_catalog(
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog {
  let embedding = try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(family(digest, 1)),
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
    implementation=ContentAddressedAot(family(digest, 2)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  try! @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries=[
    norm, embedding,
  ]).resolve(
    artifact_plan(),
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
fn entry(
  digest : @catalog.AotArtifactDigest,
  family_id : Int,
  entry_id : Int,
) -> @catalog.AotKernelEntryPoint {
  let entry_point_id = try! @catalog.AotKernelEntryPointId::from_stable_int(
    entry_id,
  )
  @catalog.AotKernelEntryPoint::new(
    family=family(digest, family_id),
    entry_point_id~,
  )
}

///|
fn operands(operation : Int, rows : Int) -> Array[@launch_contract.AotOperand] {
  let row_bytes = rows.to_int64() * 32L
  if operation == 0 {
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
        byte_count=row_bytes,
        alignment=16L,
      ),
    ]
  } else {
    [
      try! @launch_contract.AotOperand::new(
        role=ActivationInput(@plan.ActivationRef::new(0)),
        byte_count=row_bytes,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=WeightInput(@plan.TensorRef::new(1)),
        byte_count=32L,
        alignment=16L,
      ),
      try! @launch_contract.AotOperand::new(
        role=ActivationOutput(@plan.ActivationRef::new(1)),
        byte_count=row_bytes,
        alignment=16L,
      ),
    ]
  }
}

///|
fn contract_input(
  digest : @catalog.AotArtifactDigest,
  profile_id : Int,
  operation_id : Int,
  entry_id : Int,
  rows : Int,
) -> @launch_contract.AotLaunchContractInput {
  let resolved_profile = try! @launch_contract.KernelProfileId::from_stable_int(
    profile_id,
  )
  let resolved_operation = try! @plan.OperationId::new(operation_id)
  @launch_contract.AotLaunchContractInput::new(
    profile_id=resolved_profile,
    operation_id=resolved_operation,
    entry_point=entry(digest, operation_id + 1, entry_id),
    dimensions=dimensions(rows),
    operands=operands(operation_id, rows),
  )
}

///|
fn contracts(
  digest : @catalog.AotArtifactDigest,
) -> @launch_contract.LaunchContractSet {
  try! @launch_contract.admit(
    artifact_plan(),
    resolved_catalog(digest),
    [profile(1, 2), profile(2, 8)],
    [
      contract_input(digest, 2, 1, 22, 8),
      contract_input(digest, 1, 0, 11, 2),
      contract_input(digest, 2, 0, 12, 8),
      contract_input(digest, 1, 1, 21, 2),
    ],
    @launch_contract.LaunchContractLimits::new(
      max_profiles=4,
      max_contracts=8,
      max_operands_per_contract=8,
    ),
  )
}

///|
fn limits(
  max_modules? : Int = 4,
  max_entry_points? : Int = 8,
  max_module_bytes? : Int64 = 1024L,
  max_total_module_bytes? : Int64 = 4096L,
  max_function_symbol_bytes? : Int = 64,
) -> @artifact.KernelArtifactLimits {
  try! @artifact.KernelArtifactLimits::new(
    max_modules~,
    max_entry_points~,
    max_module_bytes~,
    max_total_module_bytes~,
    max_function_symbol_bytes~,
  )
}

///|
fn export_input(
  digest : @catalog.AotArtifactDigest,
  family_id : Int,
  entry_id : Int,
  symbol : String,
) -> @artifact.KernelEntryPointInput {
  @artifact.KernelEntryPointInput::new(
    entry_point=entry(digest, family_id, entry_id),
    function_symbol=symbol,
  )
}

///|
fn exports(
  digest : @catalog.AotArtifactDigest,
) -> Array[@artifact.KernelEntryPointInput] {
  [
    export_input(digest, 2, 22, "norm_prefill_8"),
    export_input(digest, 1, 11, "embedding_prefill_2"),
    export_input(digest, 1, 12, "embedding_prefill_8"),
    export_input(digest, 2, 21, "norm_prefill_2"),
  ]
}

///|
test "one immutable module backs multiple families and exact profiles" {
  let bytes = b"multi-function-cubin".to_fixedarray()
  let owned = Bytes::from_array(bytes)
  let digest = digest_for(bytes)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=bytes,
  )
  let module_inputs = [module_input]
  let entry_inputs = exports(digest)
  let bundle = @artifact.admit(
    contracts(digest),
    module_inputs,
    entry_inputs,
    limits(),
  )
  module_inputs.push(module_input)
  entry_inputs[0] = entry_inputs[1]
  bytes[0] = b'X'
  assert_eq(bundle.module_count(), 1)
  assert_eq(bundle.entry_point_count(), 4)
  assert_eq(bundle.total_module_bytes(), owned.length().to_int64())
  assert_eq(bundle.modules()[0].module_bytes(), owned)
  assert_eq(bundle.entry_points()[0].function_symbol(), "embedding_prefill_2")
  assert_eq(bundle.entry_points()[1].function_symbol(), "norm_prefill_2")
  assert_eq(bundle.entry_points()[2].function_symbol(), "embedding_prefill_8")
  assert_eq(bundle.entry_points()[3].function_symbol(), "norm_prefill_8")
  for admitted in bundle.entry_points() {
    assert_eq(admitted.module_index(), 0)
  }
  guard bundle.find_module(digest) is Some(found_module) else {
    fail("expected module lookup")
  }
  assert_eq(found_module.module_bytes(), owned)
  guard bundle.entry_point(entry(digest, 2, 22)) is Some(found_entry) else {
    fail("expected stable entry-point lookup")
  }
  assert_eq(found_entry.function_symbol(), "norm_prefill_8")
}

///|
test "module relation and digest validation fail closed" {
  let bytes = b"multi-function-cubin".to_fixedarray()
  let digest = digest_for(bytes)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=bytes,
  )
  try @artifact.admit(contracts(digest), [], exports(digest), limits()) catch {
    MissingModule(found) => assert_eq(found, digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected missing module")
  }
  try
    @artifact.admit(
      contracts(digest),
      [module_input, module_input],
      exports(digest),
      limits(),
    )
  catch {
    DuplicateModule(found) => assert_eq(found, digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected duplicate module")
  }
  let wrong = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=b"mutated".to_fixedarray(),
  )
  try
    @artifact.admit(contracts(digest), [wrong], exports(digest), limits())
  catch {
    DigestMismatch(found) => assert_eq(found, digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected digest mismatch")
  }
  let extra_bytes = b"unreferenced-cubin".to_fixedarray()
  let extra_digest = digest_for(extra_bytes)
  let extra = @artifact.KernelModuleInput::new(
    digest=extra_digest,
    module_bytes=extra_bytes,
  )
  try
    @artifact.admit(
      contracts(digest),
      [module_input, extra],
      exports(digest),
      limits(),
    )
  catch {
    UnreferencedModule(found) => assert_eq(found, extra_digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected unreferenced module")
  }
  try
    @artifact.admit(
      contracts(digest),
      [module_input],
      exports(digest),
      limits(max_module_bytes=8L),
    )
  catch {
    ModuleTooLarge(digest=found_digest, maximum=8L) =>
      assert_eq(found_digest, digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected module byte bound")
  }
}

///|
test "entry-point relations, aliases, and symbol bounds are exact" {
  let bytes = b"multi-function-cubin".to_fixedarray()
  let digest = digest_for(bytes)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=bytes,
  )
  let all = exports(digest)
  try
    @artifact.admit(contracts(digest), [module_input], all[:3], limits())
  catch {
    MissingEntryPoint(found) => assert_eq(found, entry(digest, 2, 21))
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected missing entry point")
  }
  try
    @artifact.admit(
      contracts(digest),
      [module_input],
      [all[0], all[0], all[1], all[2], all[3]],
      limits(),
    )
  catch {
    DuplicateEntryPoint(found) => assert_eq(found, entry(digest, 2, 22))
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected duplicate entry point")
  }
  let aliases = [
    export_input(digest, 1, 11, "same_symbol"),
    export_input(digest, 2, 21, "same_symbol"),
    all[2],
    all[0],
  ]
  let alias_bundle = @artifact.admit(
    contracts(digest),
    [module_input],
    aliases,
    limits(),
  )
  assert_eq(alias_bundle.entry_points()[0].function_symbol(), "same_symbol")
  assert_eq(alias_bundle.entry_points()[1].function_symbol(), "same_symbol")
  let invalid = [
    export_input(digest, 1, 11, "9invalid"),
    all[3],
    all[2],
    all[0],
  ]
  try
    @artifact.admit(contracts(digest), [module_input], invalid, limits())
  catch {
    InvalidFunctionSymbol(rule=InvalidInitial, ..) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected invalid function symbol")
  }
  let unreferenced = export_input(digest, 1, 99, "unused_export")
  try
    @artifact.admit(
      contracts(digest),
      [module_input],
      [all[0], all[1], all[2], all[3], unreferenced],
      limits(),
    )
  catch {
    UnreferencedEntryPoint(found) => assert_eq(found, entry(digest, 1, 99))
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected unreferenced entry point")
  }
  try
    @artifact.admit(
      contracts(digest),
      [module_input],
      all,
      limits(max_entry_points=3),
    )
  catch {
    EntryPointCountExceeded(maximum=3) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected entry-point count bound")
  }
}
