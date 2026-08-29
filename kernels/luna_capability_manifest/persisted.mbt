///|
priv struct LunaCapabilityDecoder {
  bytes : Bytes
  mut cursor : Int
}

///|
fn LunaCapabilityDecoder::read_u64(
  self : LunaCapabilityDecoder,
) -> UInt64 raise LunaCapabilityManifestError {
  if self.cursor < 0 || self.cursor > self.bytes.length() - 8 {
    raise Invalid(CanonicalBytes)
  }
  let mut value = 0UL
  for index in 0..<8 {
    value = value | (self.bytes[self.cursor + index].to_uint64() << (index * 8))
  }
  self.cursor += 8
  value
}

///|
fn LunaCapabilityDecoder::read_i64(
  self : LunaCapabilityDecoder,
) -> Int64 raise LunaCapabilityManifestError {
  self.read_u64().reinterpret_as_int64()
}

///|
fn LunaCapabilityDecoder::read_int(
  self : LunaCapabilityDecoder,
) -> Int raise LunaCapabilityManifestError {
  let value = self.read_i64()
  let decoded = value.to_int()
  if Int64::from_int(decoded) != value {
    raise Invalid(CanonicalBytes)
  }
  decoded
}

///|
fn LunaCapabilityDecoder::read_bool(
  self : LunaCapabilityDecoder,
) -> Bool raise LunaCapabilityManifestError {
  match self.read_int() {
    0 => false
    1 => true
    _ => raise Invalid(CanonicalBytes)
  }
}

///|
fn LunaCapabilityDecoder::read_string(
  self : LunaCapabilityDecoder,
) -> String raise LunaCapabilityManifestError {
  let length = self.read_int()
  if length < 0 || self.cursor < 0 || self.cursor > self.bytes.length() - length {
    raise Invalid(CanonicalBytes)
  }
  let value = @utf8.decode(
    Bytes::makei(length, index => self.bytes[self.cursor + index]),
  ) catch {
    _ => raise Invalid(CanonicalBytes)
  }
  self.cursor += length
  value
}

///|
fn LunaCapabilityDecoder::read_sha256(
  self : LunaCapabilityDecoder,
) -> String raise LunaCapabilityManifestError {
  let value = self.read_string()
  if !luna_capability_is_sha256(value) {
    raise Invalid(Digest)
  }
  value
}

///|
fn decode_luna_capability_shape(
  decoder : LunaCapabilityDecoder,
) -> LunaGraphShapeClass raise LunaCapabilityManifestError {
  let phase = decoder.read_int()
  let batch_rows = decoder.read_int()
  let query_tokens = decoder.read_int()
  let context_tokens = decoder.read_int()
  let token_rows = decoder.read_i64()
  if query_tokens != context_tokens ||
    token_rows != Int64::from_int(batch_rows) * Int64::from_int(query_tokens) {
    raise Invalid(ShapeClass)
  }
  let exact = match phase {
    0 =>
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows~,
        sequence_tokens=context_tokens,
      ) catch {
        _ => raise Invalid(ShapeClass)
      }
    1 =>
      @device_profile.ExactExecutionShape::full_recompute(
        batch_rows~,
        sequence_tokens=context_tokens,
      ) catch {
        _ => raise Invalid(ShapeClass)
      }
    _ => raise Unsupported(ShapeClass)
  }
  if exact.query_tokens_per_row() != query_tokens ||
    exact.token_rows() != token_rows {
    raise Invalid(ShapeClass)
  }
  LunaGraphShapeClass::from_execution_shape(exact)
}

///|
fn decode_luna_capability_numerical(
  decoder : LunaCapabilityDecoder,
) -> @luna_specialization.LunaNumericalPolicy raise LunaCapabilityManifestError {
  match decoder.read_int() {
    1 => LunaBitExactOperationOrderV1
    2 =>
      LunaToleranceV1(
        maximum_absolute_error_ppb=decoder.read_i64(),
        maximum_relative_error_ppb=decoder.read_i64(),
        requires_token_agreement=decoder.read_bool(),
      )
    _ => raise Unsupported(NumericalPolicy)
  }
}

///|
/// Decodes one approved-file candidate without signature-verification authority.
/// Publication requires full consumption, exact re-encoding, and digest equality.
pub fn admit_persisted_luna_capability_manifest(
  canonical : Bytes,
  expected_digest : LunaCapabilityManifestDigest,
  specialization : @luna_specialization.LunaSpecializationRecord,
  maximum_canonical_bytes : Int,
) -> LunaCapabilityManifestEntry raise LunaCapabilityManifestError {
  if canonical.length() <= 0 ||
    maximum_canonical_bytes <= 0 ||
    canonical.length() > maximum_canonical_bytes {
    raise Invalid(CanonicalBytes)
  }
  let decoder : LunaCapabilityDecoder = { bytes: canonical, cursor: 0 }
  if decoder.read_string() != "LUNA-CAPABILITY-MANIFEST-ENTRY-V1" ||
    decoder.read_int() != LunaCapabilityManifestVersion::v1().as_int() {
    raise Unsupported(Version)
  }
  let identity = specialization.identity()
  if decoder.read_sha256() != identity.content().as_hex() ||
    decoder.read_sha256() != identity.plan().as_hex() ||
    decoder.read_int() != specialization.operation_id().as_int() ||
    decoder.read_int() != specialization.capability().as_int() ||
    decoder.read_int() != specialization.semantic_version().as_int() ||
    decoder.read_sha256() != specialization.module_digest().sha256() ||
    decoder.read_sha256() != specialization.digest().as_hex() ||
    decoder.read_sha256() != specialization.program_digest().as_hex() ||
    decoder.read_sha256() != specialization.cuda_aot_digest().as_hex() {
    raise Mismatch(Specialization)
  }
  let architecture = LunaGpuArchitectureRange::new(
    minimum_major=decoder.read_int(),
    minimum_minor=decoder.read_int(),
    maximum_major=decoder.read_int(),
    maximum_minor=decoder.read_int(),
  )
  if !architecture.contains(specialization.target()) {
    raise Mismatch(Architecture)
  }
  let input_dtype = match decoder.read_int() {
    1 => LunaBFloat16
    _ => raise Unsupported(DType)
  }
  let accumulator_dtype = match decoder.read_int() {
    1 => LunaAccumulatorFloat32
    _ => raise Unsupported(DType)
  }
  let tensor_layout = match decoder.read_int() {
    1 => LunaRowMajorStridedV1
    _ => raise Unsupported(Layout)
  }
  let kv_layout = LunaKvLayoutContract::new(
    stable_version=decoder.read_int(),
    tokens_per_page=decoder.read_int(),
  )
  let minimum_alignment_bytes = decoder.read_i64()
  if minimum_alignment_bytes <= 0L ||
    (minimum_alignment_bytes & (minimum_alignment_bytes - 1L)) != 0L {
    raise Invalid(Alignment)
  }
  let compiler = specialization.compiler_policy()
  let compiler_identity = compiler.identity()
  if decoder.read_sha256() != compiler_identity.toolchain_digest_sha256() ||
    decoder.read_int() != compiler_identity.major() ||
    decoder.read_int() != compiler_identity.minor() ||
    decoder.read_int() != compiler_identity.patch() ||
    decoder.read_int() != compiler.optimization_level() ||
    decoder.read_bool() != compiler.fused_multiply_add() ||
    decoder.read_bool() != compiler.reassociate() ||
    decoder.read_int() != compiler.max_registers() ||
    decode_luna_capability_numerical(decoder) !=
    specialization.numerical_policy() {
    raise Mismatch(Compiler)
  }
  let shape_class = decode_luna_capability_shape(decoder)
  if shape_class !=
    LunaGraphShapeClass::from_execution_shape(specialization.profile_shape()) {
    raise Mismatch(ShapeClass)
  }
  let graph = LunaGraphCapabilityMetadata::new(
    shape_class~,
    capture_safe=decoder.read_bool(),
    graph_memory_bytes=decoder.read_i64(),
    graph_memory_alignment=decoder.read_i64(),
    eager_fallback=LunaValidatedEagerFallbackDigest::from_sha256(
      decoder.read_sha256(),
    ),
  )
  let evidence = specialization.evidence()
  for
    expected in [
      evidence.scalar_referee(),
      evidence.admitted_tensor_rows(),
      evidence.dispatch_canary(),
      evidence.benchmark(),
    ] {
    if decoder.read_sha256() != expected.as_hex() {
      raise Mismatch(Evidence)
    }
  }
  if decoder.cursor != canonical.length() {
    raise Invalid(CanonicalBytes)
  }
  let actual_digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  if actual_digest != expected_digest {
    raise Mismatch(Digest)
  }
  let entry : LunaCapabilityManifestEntry = {
    version: LunaCapabilityManifestVersion::v1(),
    specialization,
    architecture,
    input_dtype,
    accumulator_dtype,
    tensor_layout,
    kv_layout,
    minimum_alignment_bytes,
    graph,
    canonical,
    digest: actual_digest,
  }
  if encode_luna_capability_entry(
      entry.version,
      specialization,
      architecture,
      input_dtype,
      accumulator_dtype,
      tensor_layout,
      kv_layout,
      minimum_alignment_bytes,
      graph,
      maximum_canonical_bytes,
    ) !=
    canonical {
    raise Mismatch(CanonicalBytes)
  }
  entry
}
