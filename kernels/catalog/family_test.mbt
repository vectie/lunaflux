///|
test "one AOT family cannot claim incompatible operation semantics" {
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  let family_id = @catalog.AotKernelFamilyId::from_stable_int(1)
  let shared = @catalog.AotKernelFamily::new(module_digest=digest, family_id~)
  let embedding = @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(shared),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  let norm = @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=RmsNorm,
    operation_shape=NormalizationShape(width=16, epsilon=0.000001),
    capability=@plan.KernelCapabilityId::rms_norm(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(shared),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  try
    @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries=[
      embedding, norm,
    ])
  catch {
    InvalidCatalog(AotFamilySemanticMismatch) => ()
    _ => fail("unexpected catalog error")
  } noraise {
    _ => fail("expected incompatible family semantics")
  }
}
