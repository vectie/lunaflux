///|
priv struct LunaSpecializationDecoder {
  bytes : Bytes
  mut cursor : Int
}

///|
fn LunaSpecializationDecoder::read_u64(
  self : LunaSpecializationDecoder,
) -> UInt64 raise LunaSpecializationError {
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
fn LunaSpecializationDecoder::read_i64(
  self : LunaSpecializationDecoder,
) -> Int64 raise LunaSpecializationError {
  self.read_u64().reinterpret_as_int64()
}

///|
fn LunaSpecializationDecoder::read_int(
  self : LunaSpecializationDecoder,
) -> Int raise LunaSpecializationError {
  let value = self.read_i64()
  let decoded = value.to_int()
  if Int64::from_int(decoded) != value {
    raise Invalid(CanonicalBytes)
  }
  decoded
}

///|
fn LunaSpecializationDecoder::read_bool(
  self : LunaSpecializationDecoder,
) -> Bool raise LunaSpecializationError {
  match self.read_int() {
    0 => false
    1 => true
    _ => raise Invalid(CanonicalBytes)
  }
}

///|
fn LunaSpecializationDecoder::read_string(
  self : LunaSpecializationDecoder,
) -> String raise LunaSpecializationError {
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
fn LunaSpecializationDecoder::read_sha256(
  self : LunaSpecializationDecoder,
) -> String raise LunaSpecializationError {
  let value = self.read_string()
  if !luna_specialization_is_sha256(value) {
    raise Invalid(Digest)
  }
  value
}

///|
fn decode_luna_specialization_shape(
  decoder : LunaSpecializationDecoder,
) -> @plan.OperationShape raise LunaSpecializationError {
  if decoder.read_int() != 6 {
    raise Unsupported(OperationShape)
  }
  let width = decoder.read_int()
  if width <= 0 {
    raise Invalid(OperationShape)
  }
  ElementwiseShape(width~)
}

///|
fn decode_luna_profile_shape(
  decoder : LunaSpecializationDecoder,
) -> @device_profile.ExactExecutionShape raise LunaSpecializationError {
  let phase = decoder.read_int()
  let batch_rows = decoder.read_int()
  let query_tokens = decoder.read_int()
  let context_tokens = decoder.read_int()
  let token_rows = decoder.read_i64()
  if query_tokens != context_tokens ||
    token_rows != Int64::from_int(batch_rows) * Int64::from_int(query_tokens) {
    raise Invalid(ExecutionProfile)
  }
  let shape = match phase {
    0 =>
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows~,
        sequence_tokens=context_tokens,
      ) catch {
        _ => raise Invalid(ExecutionProfile)
      }
    1 =>
      @device_profile.ExactExecutionShape::full_recompute(
        batch_rows~,
        sequence_tokens=context_tokens,
      ) catch {
        _ => raise Invalid(ExecutionProfile)
      }
    _ => raise Unsupported(ExecutionProfile)
  }
  if shape.query_tokens_per_row() != query_tokens ||
    shape.token_rows() != token_rows {
    raise Invalid(ExecutionProfile)
  }
  shape
}

///|
fn decode_luna_compiler(
  decoder : LunaSpecializationDecoder,
) -> LunaCompilerPolicy raise LunaSpecializationError {
  let identity = LunaCompilerIdentity::new(
    toolchain_digest_sha256=decoder.read_sha256(),
    major=decoder.read_int(),
    minor=decoder.read_int(),
    patch=decoder.read_int(),
  )
  LunaCompilerPolicy::new(
    identity~,
    optimization_level=decoder.read_int(),
    fused_multiply_add=decoder.read_bool(),
    reassociate=decoder.read_bool(),
    max_registers=decoder.read_int(),
  )
}

///|
fn decode_luna_numerical(
  decoder : LunaSpecializationDecoder,
) -> LunaNumericalPolicy raise LunaSpecializationError {
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
/// Parses one bounded specialization without granting compiler/file authority.
/// Retains the supplied label without checksum or re-encoding replay. Execution
/// semantics must still match the model, profile and catalog before use.
pub fn admit_persisted_luna_specialization(
  canonical : Bytes,
  expected_digest : LunaSpecializationDigest,
  limits : LunaSpecializationLimits,
) -> LunaSpecializationRecord raise LunaSpecializationError {
  if canonical.length() <= 0 || canonical.length() > limits.max_canonical_bytes {
    raise Invalid(CanonicalBytes)
  }
  let decoder : LunaSpecializationDecoder = { bytes: canonical, cursor: 0, }
  if decoder.read_string() != "LUNA-SPECIALIZATION-V1" {
    raise Unsupported(CanonicalBytes)
  }
  // Decoded identity is an untrusted claim until exact admission against a
  // rebuilt ModelPlan identity.
  let content = @spec.ContentDigest::from_sha256(decoder.read_sha256()) catch {
    _ => raise Invalid(Identity)
  }
  let plan = @spec.PlanDigest::from_sha256(decoder.read_sha256()) catch {
    _ => raise Invalid(Identity)
  }
  let identity = @spec.ModelIdentity::new(content, plan) catch {
    _ => raise Invalid(Identity)
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=decoder.read_int(),
    compute_minor=decoder.read_int(),
    supports_bf16=decoder.read_bool(),
    supports_cublas_lt=decoder.read_bool(),
  ) catch {
    _ => raise Invalid(Target)
  }
  if decoder.read_int() != @catalog.CatalogVersion::v3().as_int() {
    raise Unsupported(CatalogVersion)
  }
  let catalog_version = @catalog.CatalogVersion::v3()
  let operation_id = @plan.OperationId::new(decoder.read_int()) catch {
    _ => raise Invalid(Operation)
  }
  let capability = @plan.KernelCapabilityId::from_stable_int(decoder.read_int()) catch {
    _ => raise Invalid(OperationBinding)
  }
  if capability != @plan.KernelCapabilityId::residual_add() {
    raise Unsupported(OperationBinding)
  }
  let semantic_version = @catalog.KernelSemanticVersion::from_stable_int(
    decoder.read_int(),
  ) catch {
    _ => raise Invalid(OperationBinding)
  }
  let module_digest = @catalog.AotArtifactDigest::from_sha256(
    decoder.read_sha256(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let entry_point_id = LunaSpecializedEntryPointId::from_stable_int(
    decoder.read_int(),
  )
  let operation_shape = decode_luna_specialization_shape(decoder)
  let profile_shape = decode_luna_profile_shape(decoder)
  let workspace_bytes = decoder.read_i64()
  let workspace_alignment = decoder.read_i64()
  if workspace_bytes < 0L ||
    workspace_alignment <= 0L ||
    (workspace_alignment & (workspace_alignment - 1L)) != 0L {
    raise Invalid(Workspace)
  }
  let rope_theta_bits = decoder.read_u64()
  let rms_epsilon_bits = decoder.read_u64()
  if decoder.read_int() != 1 {
    raise Unsupported(Quantization)
  }
  let compiler = decode_luna_compiler(decoder)
  let numerical = decode_luna_numerical(decoder)
  validate_luna_numerical_policy(compiler, numerical)
  let program_digest = @luna_tile_ir.LunaTileProgramDigest::from_sha256(
    decoder.read_sha256(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let cuda_aot_digest = @luna_tile_ir.LunaTileCudaAotDigest::from_sha256(
    decoder.read_sha256(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let region_count = decoder.read_int()
  if region_count <= 0 || region_count > limits.max_bound_regions {
    raise Invalid(TensorLayout)
  }
  let regions : Array[LunaSpecializedRegion] = []
  for _ in 0..<region_count {
    let space = match decoder.read_int() {
      1 => LunaWeightArena
      2 => LunaTokenInput
      3 => LunaActivationWorkspaceArena
      _ => raise Invalid(TensorLayout)
    }
    regions.push(
      luna_specialization_region(
        space,
        decoder.read_int(),
        decoder.read_i64(),
        decoder.read_i64(),
        decoder.read_i64(),
      ),
    )
  }
  let evidence = LunaSpecializationEvidence::new(
    scalar_referee=LunaSpecializationEvidenceDigest::from_sha256(
      decoder.read_sha256(),
    ),
    admitted_tensor_rows=LunaSpecializationEvidenceDigest::from_sha256(
      decoder.read_sha256(),
    ),
    dispatch_canary=LunaSpecializationEvidenceDigest::from_sha256(
      decoder.read_sha256(),
    ),
    benchmark=LunaSpecializationEvidenceDigest::from_sha256(
      decoder.read_sha256(),
    ),
  )
  if decoder.cursor != canonical.length() {
    raise Invalid(CanonicalBytes)
  }
  let record : LunaSpecializationRecord = {
    identity,
    target,
    catalog_version,
    operation_id,
    operation_kind: ResidualAdd,
    operation_shape,
    capability,
    semantic_version,
    module_digest,
    entry_point_id,
    compiler,
    numerical,
    quantization: LunaUnquantizedBf16V1,
    profile_shape,
    regions: ReadOnlyArray::from_array(regions),
    workspace_bytes,
    workspace_alignment,
    rope_theta_bits,
    rms_epsilon_bits,
    program_digest,
    cuda_aot_digest,
    evidence,
    canonical,
    digest: expected_digest,
  }
  record
}

///|
/// Rebinds a persisted record to exact production-derived immutable evidence.
pub fn readmit_persisted_luna_specialization(
  record : LunaSpecializationRecord,
  model_plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  profile : @device_profile.DeviceExecutionProfile,
  catalog : @catalog.ResolvedKernelCatalog,
) -> LunaSpecializationRecord raise LunaSpecializationError {
  if record.identity != model_plan.identity() ||
    record.identity != static_plan.identity() ||
    record.identity != profile.identity() ||
    record.identity != catalog.model_identity() {
    raise Mismatch(Identity)
  }
  if record.target != static_plan.target() ||
    record.target != profile.target() ||
    record.target != catalog.device_target() {
    raise Mismatch(Target)
  }
  guard luna_specialization_plan_operation(model_plan, record.operation_id)
    is Some(operation) &&
    luna_specialization_static_operation(static_plan, record.operation_id)
    is Some(static_operation) &&
    profile.operation(record.operation_id) is Some(operation_profile) &&
    luna_specialization_binding(catalog, record.operation_id) is Some(binding) else {
    raise Mismatch(Operation)
  }
  let module_digest = match binding.implementation() {
    ContentAddressedAot(family) => family.module_digest()
    _ => raise Unsupported(OperationBinding)
  }
  let expected_regions = collect_luna_specialization_regions(
    operation_profile,
    record.regions.length(),
  )
  let (rope_theta_bits, rms_epsilon_bits) = collect_luna_semantic_bits(
    model_plan,
  )
  if operation.kind() != ResidualAdd ||
    operation.kind() != record.operation_kind ||
    operation.shape() != record.operation_shape ||
    operation.required_capability() != record.capability ||
    static_operation.kind() != record.operation_kind ||
    static_operation.shape() != record.operation_shape ||
    static_operation.binding() != binding ||
    binding.semantic_version() != record.semantic_version ||
    module_digest != record.module_digest ||
    binding.workspace_bytes() != record.workspace_bytes ||
    binding.workspace_alignment() != record.workspace_alignment ||
    profile.shape() != record.profile_shape ||
    expected_regions != record.regions ||
    rope_theta_bits != record.rope_theta_bits ||
    rms_epsilon_bits != record.rms_epsilon_bits {
    raise Mismatch(OperationBinding)
  }
  record
}
