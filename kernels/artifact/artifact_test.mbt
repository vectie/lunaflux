///|
fn artifact_test_identity() -> @spec.ModelIdentity {
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
fn artifact_test_plan() -> @plan.ModelPlan {
  let embedding = try! @plan.PlanOperation::new(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let projection = try! @plan.PlanOperation::new(
    id=@plan.OperationId::new(1),
    kind=QkvProjection,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
      Tensor(@plan.TensorRef::new(2)),
      Tensor(@plan.TensorRef::new(3)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=ProjectionShape(input_width=16, output_width=32),
    required_capability=@plan.KernelCapabilityId::qkv_projection(),
    workspace_bytes=0L,
  )
  try! @plan.ModelPlan::new(
    identity=artifact_test_identity(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[embedding, projection],
    final_output=@plan.ActivationRef::new(1),
  )
}

///|
fn artifact_test_digest(
  bytes : FixedArray[Byte],
) -> @catalog.AotArtifactDigest raise @catalog.CatalogError {
  @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn artifact_test_catalog_entry(
  digest : @catalog.AotArtifactDigest,
  operation_kind : @plan.OperationKind,
) -> @catalog.CatalogEntry raise {
  let shape : @plan.OperationShape = match operation_kind {
    Embedding => EmbeddingShape(vocabulary_size=128, width=16)
    QkvProjection => ProjectionShape(input_width=16, output_width=32)
    _ => abort("unsupported test operation")
  }
  let capability = match operation_kind {
    Embedding => @plan.KernelCapabilityId::embedding_lookup()
    QkvProjection => @plan.KernelCapabilityId::qkv_projection()
    _ => abort("unsupported test operation")
  }
  @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind~,
    operation_shape=shape,
    capability~,
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(digest),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
}

///|
fn artifact_test_resolved(
  first : @catalog.AotArtifactDigest,
  second : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let catalog = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries=[
      artifact_test_catalog_entry(second, QkvProjection),
      artifact_test_catalog_entry(first, Embedding),
    ],
  )
  catalog.resolve(
    artifact_test_plan(),
    @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn artifact_test_limits(
  max_artifacts? : Int = 8,
  max_module_bytes? : Int64 = 1024L,
  max_total_module_bytes? : Int64 = 4096L,
  max_function_symbol_bytes? : Int = 64,
) -> @artifact.KernelArtifactLimits raise @artifact.KernelArtifactError {
  @artifact.KernelArtifactLimits::new(
    max_artifacts~,
    max_module_bytes~,
    max_total_module_bytes~,
    max_function_symbol_bytes~,
  )
}

///|
fn artifact_test_input(
  bytes : FixedArray[Byte],
  symbol : String,
) -> @artifact.KernelArtifactInput raise @catalog.CatalogError {
  @artifact.KernelArtifactInput::new(
    digest=artifact_test_digest(bytes),
    module_bytes=bytes,
    function_symbol=symbol,
  )
}

///|
test "admission verifies exact modules and orders by first resolved use" {
  let first_bytes = b"first-cubin".to_fixedarray()
  let second_bytes = b"second-cubin".to_fixedarray()
  let first_owned = Bytes::from_array(first_bytes)
  let second_owned = Bytes::from_array(second_bytes)
  let first_digest = artifact_test_digest(first_bytes)
  let second_digest = artifact_test_digest(second_bytes)
  let first = artifact_test_input(first_bytes, "_Z16embedding_kernelv")
  let second = artifact_test_input(second_bytes, "qkv_kernel_v1")
  let inputs = [second, first]
  let bundle = @artifact.admit(
    artifact_test_resolved(first_digest, second_digest),
    inputs,
    artifact_test_limits(),
  )
  inputs[0] = first
  first_bytes[0] = b'X'
  assert_eq(bundle.length(), 2)
  assert_eq(bundle.model_identity(), artifact_test_identity())
  let resolved = artifact_test_resolved(first_digest, second_digest)
  assert_eq(bundle.device_target(), resolved.device_target())
  assert_eq(bundle.catalog_version(), resolved.catalog_version())
  assert_eq(
    bundle.total_module_bytes(),
    (first_bytes.length() + second_bytes.length()).to_int64(),
  )
  assert_eq(bundle.entries()[0].digest(), first_digest)
  assert_eq(bundle.entries()[0].module_bytes(), first_owned)
  assert_eq(bundle.entries()[0].function_symbol(), "_Z16embedding_kernelv")
  assert_eq(bundle.entries()[1].digest(), second_digest)
  assert_eq(bundle.entries()[1].module_bytes(), second_owned)
  assert_eq(bundle.entries()[1].function_symbol(), "qkv_kernel_v1")
  guard bundle.entry(first_digest) is Some(found) else {
    fail("expected deterministic digest lookup")
  }
  assert_eq(found.module_bytes(), first_owned)
}

///|
test "digest mismatch and duplicate artifacts fail closed" {
  let first_bytes = b"first-cubin".to_fixedarray()
  let second_bytes = b"second-cubin".to_fixedarray()
  let first_digest = artifact_test_digest(first_bytes)
  let second_digest = artifact_test_digest(second_bytes)
  let resolved = artifact_test_resolved(first_digest, second_digest)
  let wrong = @artifact.KernelArtifactInput::new(
    digest=first_digest,
    module_bytes=b"mutated-cubin".to_fixedarray(),
    function_symbol="embedding_kernel",
  )
  let second = artifact_test_input(second_bytes, "qkv_kernel")
  try @artifact.admit(resolved, [wrong, second], artifact_test_limits()) catch {
    DigestMismatch(found) => assert_eq(found, first_digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected digest mismatch")
  }
  let first = artifact_test_input(first_bytes, "embedding_kernel")
  try
    @artifact.admit(resolved, [first, first, second], artifact_test_limits())
  catch {
    DuplicateArtifact(found) => assert_eq(found, first_digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected duplicate artifact")
  }
}

///|
test "CUDA symbols are bounded conservative ASCII identifiers" {
  let first_bytes = b"first-cubin".to_fixedarray()
  let second_bytes = b"second-cubin".to_fixedarray()
  let first_digest = artifact_test_digest(first_bytes)
  let resolved = artifact_test_resolved(
    first_digest,
    artifact_test_digest(second_bytes),
  )
  let second = artifact_test_input(second_bytes, "qkv_kernel")
  let invalid_symbols : Array[(String, @artifact.FunctionSymbolRule)] = [
    ("", Empty),
    ("9kernel", InvalidInitial),
    ("kernel-name", InvalidContinuation),
    ("kernél", NonAscii),
  ]
  for item in invalid_symbols {
    let (symbol, expected) = item
    let invalid = @artifact.KernelArtifactInput::new(
      digest=first_digest,
      module_bytes=first_bytes,
      function_symbol=symbol,
    )
    try
      @artifact.admit(resolved, [invalid, second], artifact_test_limits())
    catch {
      InvalidFunctionSymbol(digest=found, rule~) => {
        assert_eq(found, first_digest)
        assert_eq(rule, expected)
      }
      _ => fail("unexpected artifact error")
    } noraise {
      _ => fail("expected invalid symbol")
    }
  }
  let too_long = @artifact.KernelArtifactInput::new(
    digest=first_digest,
    module_bytes=first_bytes,
    function_symbol="kernel",
  )
  try
    @artifact.admit(
      resolved,
      [too_long, second],
      artifact_test_limits(max_function_symbol_bytes=5),
    )
  catch {
    InvalidFunctionSymbol(rule=TooLong, ..) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected bounded symbol failure")
  }
}

///|
test "missing and unreferenced catalog artifacts are rejected" {
  let first_bytes = b"first-cubin".to_fixedarray()
  let second_bytes = b"second-cubin".to_fixedarray()
  let first = artifact_test_input(first_bytes, "embedding_kernel")
  let second_digest = artifact_test_digest(second_bytes)
  let resolved = artifact_test_resolved(
    artifact_test_digest(first_bytes),
    second_digest,
  )
  try @artifact.admit(resolved, [first], artifact_test_limits()) catch {
    MissingArtifact(found) => assert_eq(found, second_digest)
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected missing artifact")
  }
  let second = artifact_test_input(second_bytes, "qkv_kernel")
  let extra_bytes = b"unreferenced-cubin".to_fixedarray()
  let extra = artifact_test_input(extra_bytes, "unused_kernel")
  try
    @artifact.admit(resolved, [first, second, extra], artifact_test_limits())
  catch {
    UnreferencedArtifact(found) =>
      assert_eq(found, artifact_test_digest(extra_bytes))
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected unreferenced artifact")
  }
}

///|
test "count module total and representability limits precede ownership" {
  let first_bytes = b"12345678".to_fixedarray()
  let second_bytes = b"abcdefgh".to_fixedarray()
  let first = artifact_test_input(first_bytes, "first_kernel")
  let second = artifact_test_input(second_bytes, "second_kernel")
  let resolved = artifact_test_resolved(
    artifact_test_digest(first_bytes),
    artifact_test_digest(second_bytes),
  )
  try
    @artifact.admit(
      resolved,
      [first, second],
      artifact_test_limits(max_artifacts=1),
    )
  catch {
    ArtifactCountExceeded(maximum=1) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected artifact count bound")
  }
  try
    @artifact.admit(
      resolved,
      [first, second],
      artifact_test_limits(max_module_bytes=7L),
    )
  catch {
    ModuleTooLarge(maximum=7L, ..) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected module bound")
  }
  try
    @artifact.admit(
      resolved,
      [first, second],
      artifact_test_limits(max_total_module_bytes=15L),
    )
  catch {
    TotalModuleBytesExceeded(maximum=15L) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected total bound")
  }
  try
    @artifact.KernelArtifactLimits::new(
      max_artifacts=1,
      max_module_bytes=1L,
      max_total_module_bytes=9223372036854775807L,
      max_function_symbol_bytes=1,
    )
  catch {
    InvalidLimit(TotalModuleBytes) => ()
    _ => fail("unexpected artifact error")
  } noraise {
    _ => fail("expected non-representable total limit rejection")
  }
}
