///|
fn test_identity() -> @spec.ModelIdentity {
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
fn test_operation(workspace_bytes : Int64) -> @plan.PlanOperation {
  try! @plan.PlanOperation::new(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes~,
  )
}

///|
fn test_plan(workspace_limit : Int64) -> @plan.ModelPlan {
  try! @plan.ModelPlan::new(
    identity=test_identity(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(
      max_temporary_bytes=workspace_limit,
    ),
    operations=[test_operation(0L)],
    final_output=@plan.ActivationRef::new(0),
  )
}

///|
fn output_projection_operation(workspace_bytes : Int64) -> @plan.PlanOperation {
  try! @plan.PlanOperation::new(
    id=@plan.OperationId::new(1),
    kind=OutputProjection,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=ProjectionShape(input_width=16, output_width=16),
    required_capability=@plan.KernelCapabilityId::dense_projection(),
    workspace_bytes~,
  )
}

///|
fn projection_plan(
  workspace_limit : Int64,
  operation_workspace? : Int64 = 0L,
) -> @plan.ModelPlan {
  try! @plan.ModelPlan::new(
    identity=test_identity(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(
      max_temporary_bytes=workspace_limit,
    ),
    operations=[
      test_operation(0L),
      output_projection_operation(operation_workspace),
    ],
    final_output=@plan.ActivationRef::new(1),
  )
}

///|
fn test_target(
  supports_bf16? : Bool = true,
  supports_cublas_lt? : Bool = true,
) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16~,
    supports_cublas_lt~,
  )
}

///|
fn aot_family(
  digest : @catalog.AotArtifactDigest,
  stable_id? : Int = 1,
) -> @catalog.AotKernelFamily {
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(stable_id)
  @catalog.AotKernelFamily::new(module_digest=digest, family_id~)
}

///|
fn aot_entry(
  workspace_bytes? : Int64 = 31L,
  workspace_alignment? : Int64 = 16L,
) -> @catalog.CatalogEntry {
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(aot_family(digest)),
    workspace_bytes~,
    workspace_alignment~,
  )
}

///|
fn vendor_projection_entry(
  workspace_bytes? : Int64 = 0L,
  workspace_alignment? : Int64 = 1L,
) -> @catalog.CatalogEntry {
  try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=OutputProjection,
    operation_shape=ProjectionShape(input_width=16, output_width=16),
    capability=@plan.KernelCapabilityId::dense_projection(),
    compute_major=8,
    compute_minor=0,
    implementation=VendorBf16GemmImplicitHeuristicV1,
    workspace_bytes~,
    workspace_alignment~,
  )
}

///|
fn catalog_of(entries : Array[@catalog.CatalogEntry]) -> @catalog.KernelCatalog {
  try! @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v1(),
    entries~,
  )
}

///|
test "catalog resolves one exact immutable binding" {
  let entries = [aot_entry()]
  let catalog = catalog_of(entries)
  entries.push(aot_entry())
  let resolved = catalog.resolve(test_plan(32L), test_target())
  assert_eq(resolved.bindings().length(), 1)
  assert_eq(resolved.bindings()[0].operation_id().as_int(), 0)
  assert_eq(resolved.bindings()[0].workspace_bytes(), 32L)
  assert_eq(resolved.max_workspace_bytes(), 32L)
  assert_eq(resolved.max_workspace_alignment(), 16L)
  assert_eq(resolved.model_identity(), test_identity())
  assert_eq(resolved.device_target(), test_target())
  guard resolved.bindings()[0].implementation() is ContentAddressedAot(_) else {
    fail("expected selected content-addressed identity")
  }
}

///|
test "AOT identity is content addressed and exact" {
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  let entry = @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(aot_family(digest)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  let resolved = catalog_of([entry]).resolve(
    test_plan(0L),
    test_target(supports_cublas_lt=false),
  )
  guard resolved.bindings()[0].implementation() is ContentAddressedAot(found) else {
    fail("expected selected AOT identity")
  }
  assert_eq(found.module_digest().sha256(), digest.sha256())
  assert_eq(found.family_id().as_int(), 1)
}

///|
test "missing semantic support fails closed" {
  let entry = @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=RmsNorm,
    operation_shape=NormalizationShape(width=16, epsilon=0.000001),
    capability=@plan.KernelCapabilityId::rms_norm(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(
      aot_family(
        @catalog.AotArtifactDigest::from_sha256(
          "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        ),
      ),
    ),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  try catalog_of([entry]).resolve(test_plan(0L), test_target()) catch {
    MissingKernel(id) => assert_eq(id.as_int(), 0)
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected missing kernel")
  }
}

///|
test "architecture mismatch and unsupported vendor path are distinct" {
  let catalog = catalog_of([aot_entry()])
  let wrong_arch = @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try catalog.resolve(test_plan(32L), wrong_arch) catch {
    IncompatibleTarget(id) => assert_eq(id.as_int(), 0)
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected incompatible target")
  }
  try
    catalog_of([
      aot_entry(workspace_bytes=0L, workspace_alignment=1L),
      vendor_projection_entry(),
    ]).resolve(projection_plan(0L), test_target(supports_cublas_lt=false))
  catch {
    IncompatibleImplementation(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected incompatible implementation")
  }
}

///|
test "duplicate exact matches are rejected as ambiguous regardless of order" {
  let first = aot_entry()
  let second_digest = @catalog.AotArtifactDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let second = @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(aot_family(second_digest)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  for entries in [[first, second], [second, first]] {
    try catalog_of(entries).resolve(test_plan(32L), test_target()) catch {
      AmbiguousKernel(id) => assert_eq(id.as_int(), 0)
      _ => fail("unexpected catalog error")
    } noraise {
      _ => fail("expected ambiguous kernel")
    }
  }
}

///|
test "workspace rounding rejects overflow and limit excess" {
  try
    catalog_of([
      aot_entry(workspace_bytes=9223372036854775807L, workspace_alignment=2L),
    ]).resolve(test_plan(9223372036854775807L), test_target())
  catch {
    WorkspaceSizeOverflow(id) => assert_eq(id.as_int(), 0)
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected workspace overflow")
  }
  try catalog_of([aot_entry()]).resolve(test_plan(31L), test_target()) catch {
    WorkspaceLimitExceeded(id) => assert_eq(id.as_int(), 0)
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected workspace limit failure")
  }
}

///|
test "entry workspace must cover documented operation workspace" {
  try
    catalog_of([
      aot_entry(workspace_bytes=0L, workspace_alignment=1L),
      vendor_projection_entry(workspace_bytes=32L, workspace_alignment=16L),
    ]).resolve(projection_plan(64L, operation_workspace=64L), test_target())
  catch {
    WorkspaceRequirementMismatch(id) => assert_eq(id.as_int(), 1)
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected operation workspace mismatch")
  }
}

///|
test "implicit vendor GEMM is legal only for representable dense projections" {
  try
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=Embedding,
      operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
      capability=@plan.KernelCapabilityId::embedding_lookup(),
      compute_major=8,
      compute_minor=0,
      implementation=VendorBf16GemmImplicitHeuristicV1,
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
  catch {
    InvalidCatalog(InvalidVendorImplementationKind) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected illegal vendor implementation kind")
  }
  try
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=QkvProjection,
      operation_shape=ProjectionShape(input_width=16, output_width=32),
      capability=@plan.KernelCapabilityId::qkv_projection(),
      compute_major=8,
      compute_minor=0,
      implementation=VendorBf16GemmImplicitHeuristicV1,
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
  catch {
    InvalidCatalog(InvalidVendorImplementationKind) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected QKV to require an AOT ABI")
  }
}

///|
test "entry shape and capability must match its operation semantics" {
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  try
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=Embedding,
      operation_shape=ElementwiseShape(width=16),
      capability=@plan.KernelCapabilityId::embedding_lookup(),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(aot_family(digest)),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
  catch {
    InvalidCatalog(OperationShapeMismatch) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected operation shape mismatch")
  }
  try
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=Embedding,
      operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
      capability=@plan.KernelCapabilityId::residual_add(),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(aot_family(digest)),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
  catch {
    InvalidCatalog(OperationCapabilityMismatch) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected operation capability mismatch")
  }
}

///|
test "version, digest, alignment, and target validation fail closed" {
  try
    @catalog.KernelCatalog::new(
      version=@catalog.CatalogVersion::from_stable_int(2),
      entries=[aot_entry()],
    )
  catch {
    InvalidCatalog(UnsupportedCatalogVersion) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected unsupported catalog version")
  }
  try @catalog.AotArtifactDigest::from_sha256("ABC") catch {
    InvalidCatalog(InvalidArtifactDigest) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected invalid digest")
  }
  try @catalog.AotKernelFamilyId::from_stable_int(0) catch {
    InvalidCatalog(InvalidAotFamilyId) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected invalid AOT family identity")
  }
  try @catalog.AotKernelEntryPointId::from_stable_int(-1) catch {
    InvalidCatalog(InvalidAotEntryPointId) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected invalid AOT entry-point identity")
  }
  try
    @catalog.CatalogEntry::new(
      semantic_version=@catalog.KernelSemanticVersion::v1(),
      operation_kind=Embedding,
      operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
      capability=@plan.KernelCapabilityId::embedding_lookup(),
      compute_major=8,
      compute_minor=0,
      implementation=ContentAddressedAot(
        aot_family(
          @catalog.AotArtifactDigest::from_sha256(
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
          ),
        ),
      ),
      workspace_bytes=0L,
      workspace_alignment=3L,
    )
  catch {
    InvalidCatalog(InvalidWorkspaceAlignment) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected invalid alignment")
  }
  try
    @catalog.DeviceTarget::new(
      compute_major=0,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
    )
  catch {
    InvalidCatalog(InvalidComputeCapability) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected invalid compute capability")
  }
}
