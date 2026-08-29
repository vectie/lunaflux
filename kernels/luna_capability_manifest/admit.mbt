///|
pub fn LunaCapabilityManifestEntry::from_specialization(
  specialization~ : @luna_specialization.LunaSpecializationRecord,
  architecture~ : LunaGpuArchitectureRange,
  input_dtype~ : LunaCapabilityDType,
  accumulator_dtype~ : LunaAccumulatorDType,
  tensor_layout~ : LunaTensorLayoutContract,
  kv_layout~ : LunaKvLayoutContract,
  minimum_alignment_bytes~ : Int64,
  graph~ : LunaGraphCapabilityMetadata,
  maximum_canonical_bytes~ : Int,
) -> LunaCapabilityManifestEntry raise LunaCapabilityManifestError {
  if maximum_canonical_bytes <= 0 {
    raise Invalid(CanonicalBytes)
  }
  if !architecture.contains(specialization.target()) {
    raise Mismatch(Architecture)
  }
  if input_dtype != LunaBFloat16 ||
    accumulator_dtype != LunaAccumulatorFloat32 ||
    specialization.quantization_policy() != LunaUnquantizedBf16V1 {
    raise Unsupported(DType)
  }
  if minimum_alignment_bytes <= 0L ||
    (minimum_alignment_bytes & (minimum_alignment_bytes - 1L)) != 0L {
    raise Invalid(Alignment)
  }
  for region in specialization.regions() {
    if region.alignment_bytes() < minimum_alignment_bytes {
      raise Mismatch(Alignment)
    }
  }
  if graph.shape_class !=
    LunaGraphShapeClass::from_execution_shape(specialization.profile_shape()) {
    raise Mismatch(ShapeClass)
  }
  let version = LunaCapabilityManifestVersion::v1()
  let canonical = encode_luna_capability_entry(
    version, specialization, architecture, input_dtype, accumulator_dtype, tensor_layout,
    kv_layout, minimum_alignment_bytes, graph, maximum_canonical_bytes,
  )
  let digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  {
    version,
    specialization,
    architecture,
    input_dtype,
    accumulator_dtype,
    tensor_layout,
    kv_layout,
    minimum_alignment_bytes,
    graph,
    canonical,
    digest,
  }
}

///|
pub fn admit_luna_external_approved_capability(
  entry : LunaCapabilityManifestEntry,
  envelope : LunaDetachedSignatureEnvelope,
  approval : LunaAuthenticatedExternalApproval,
) -> LunaAdmittedCapabilityManifestEntry raise LunaCapabilityManifestError {
  if entry.version != LunaCapabilityManifestVersion::v1() {
    raise Unsupported(Version)
  }
  let actual_digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(entry.canonical)),
  )
  if entry.digest != actual_digest || approval.manifest_digest != actual_digest {
    raise Mismatch(Digest)
  }
  if envelope.algorithm != approval.algorithm ||
    envelope.key_identity != approval.key_identity ||
    envelope.signature_digest != approval.signature_digest {
    raise Mismatch(ExternalApproval)
  }
  {
    entry,
    approved_source: approval.approved_source,
    key_identity: approval.key_identity,
    signature_digest: approval.signature_digest,
  }
}
