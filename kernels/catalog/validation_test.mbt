///|
test "version, digest, alignment, and target validation fail closed" {
  try
    @catalog.KernelCatalog::new(
      version=@catalog.CatalogVersion::from_stable_int(5),
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
