///|
priv struct LunaCapabilityEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn LunaCapabilityEncoder::append_byte(
  self : LunaCapabilityEncoder,
  value : Byte,
) -> Unit raise LunaCapabilityManifestError {
  if self.bytes.length() >= self.maximum {
    raise Invalid(CanonicalBytes)
  }
  self.bytes.push(value)
}

///|
fn LunaCapabilityEncoder::append_u64(
  self : LunaCapabilityEncoder,
  value : UInt64,
) -> Unit raise LunaCapabilityManifestError {
  for byte in value.to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn LunaCapabilityEncoder::append_int(
  self : LunaCapabilityEncoder,
  value : Int,
) -> Unit raise LunaCapabilityManifestError {
  self.append_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn LunaCapabilityEncoder::append_i64(
  self : LunaCapabilityEncoder,
  value : Int64,
) -> Unit raise LunaCapabilityManifestError {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn LunaCapabilityEncoder::append_string(
  self : LunaCapabilityEncoder,
  value : StringView,
) -> Unit raise LunaCapabilityManifestError {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
}

///|
fn LunaCapabilityEncoder::append_numerical(
  self : LunaCapabilityEncoder,
  value : @luna_specialization.LunaNumericalPolicy,
) -> Unit raise LunaCapabilityManifestError {
  match value {
    LunaBitExactOperationOrderV1 => self.append_int(1)
    LunaToleranceV1(
      maximum_absolute_error_ppb~,
      maximum_relative_error_ppb~,
      requires_token_agreement~
    ) => {
      self.append_int(2)
      self.append_i64(maximum_absolute_error_ppb)
      self.append_i64(maximum_relative_error_ppb)
      self.append_int(if requires_token_agreement { 1 } else { 0 })
    }
  }
}

///|
fn encode_luna_capability_entry(
  version : LunaCapabilityManifestVersion,
  specialization : @luna_specialization.LunaSpecializationRecord,
  architecture : LunaGpuArchitectureRange,
  input_dtype : LunaCapabilityDType,
  accumulator_dtype : LunaAccumulatorDType,
  tensor_layout : LunaTensorLayoutContract,
  kv_layout : LunaKvLayoutContract,
  minimum_alignment_bytes : Int64,
  graph : LunaGraphCapabilityMetadata,
  maximum : Int,
) -> Bytes raise LunaCapabilityManifestError {
  let encoder : LunaCapabilityEncoder = { bytes: [], maximum }
  encoder.append_string("LUNA-CAPABILITY-MANIFEST-ENTRY-V1")
  encoder.append_int(version.as_int())
  let identity = specialization.identity()
  encoder.append_string(identity.content().as_hex())
  encoder.append_string(identity.plan().as_hex())
  encoder.append_int(specialization.operation_id().as_int())
  encoder.append_int(specialization.capability().as_int())
  encoder.append_int(specialization.semantic_version().as_int())
  encoder.append_string(specialization.module_digest().sha256())
  encoder.append_string(specialization.digest().as_hex())
  encoder.append_string(specialization.program_digest().as_hex())
  encoder.append_string(specialization.cuda_aot_digest().as_hex())
  encoder.append_int(architecture.minimum_major)
  encoder.append_int(architecture.minimum_minor)
  encoder.append_int(architecture.maximum_major)
  encoder.append_int(architecture.maximum_minor)
  encoder.append_int(
    match input_dtype {
      LunaBFloat16 => 1
      LunaFloat16 => 2
      LunaFloat32 => 3
    },
  )
  encoder.append_int(
    match accumulator_dtype {
      LunaAccumulatorFloat32 => 1
    },
  )
  encoder.append_int(
    match tensor_layout {
      LunaRowMajorStridedV1 => 1
    },
  )
  encoder.append_int(kv_layout.stable_version)
  encoder.append_int(kv_layout.tokens_per_page)
  encoder.append_i64(minimum_alignment_bytes)
  let compiler = specialization.compiler_policy()
  let compiler_identity = compiler.identity()
  encoder.append_string(compiler_identity.toolchain_digest_sha256())
  encoder.append_int(compiler_identity.major())
  encoder.append_int(compiler_identity.minor())
  encoder.append_int(compiler_identity.patch())
  encoder.append_int(compiler.optimization_level())
  encoder.append_int(if compiler.fused_multiply_add() { 1 } else { 0 })
  encoder.append_int(if compiler.reassociate() { 1 } else { 0 })
  encoder.append_int(compiler.max_registers())
  encoder.append_numerical(specialization.numerical_policy())
  let shape = graph.shape_class
  encoder.append_int(shape.phase.as_stable_int())
  encoder.append_int(shape.batch_rows)
  encoder.append_int(shape.query_tokens_per_row)
  encoder.append_int(shape.context_tokens)
  encoder.append_i64(shape.token_rows)
  encoder.append_int(if graph.capture_safe { 1 } else { 0 })
  encoder.append_i64(graph.graph_memory_bytes)
  encoder.append_i64(graph.graph_memory_alignment)
  encoder.append_string(graph.eager_fallback.as_hex())
  let evidence = specialization.evidence()
  for
    digest in [
      evidence.scalar_referee(),
      evidence.admitted_tensor_rows(),
      evidence.dispatch_canary(),
      evidence.benchmark(),
    ] {
    encoder.append_string(digest.as_hex())
  }
  Bytes::from_array(encoder.bytes)
}
