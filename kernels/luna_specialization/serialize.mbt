///|
priv struct LunaSpecializationEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn LunaSpecializationEncoder::append_byte(
  self : LunaSpecializationEncoder,
  value : Byte,
) -> Unit raise LunaSpecializationError {
  if self.bytes.length() >= self.maximum {
    raise Invalid(CanonicalBytes)
  }
  self.bytes.push(value)
}

///|
fn LunaSpecializationEncoder::append_u64(
  self : LunaSpecializationEncoder,
  value : UInt64,
) -> Unit raise LunaSpecializationError {
  for byte in value.to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn LunaSpecializationEncoder::append_int(
  self : LunaSpecializationEncoder,
  value : Int,
) -> Unit raise LunaSpecializationError {
  self.append_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn LunaSpecializationEncoder::append_i64(
  self : LunaSpecializationEncoder,
  value : Int64,
) -> Unit raise LunaSpecializationError {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn LunaSpecializationEncoder::append_string(
  self : LunaSpecializationEncoder,
  value : StringView,
) -> Unit raise LunaSpecializationError {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
}

///|
fn operation_shape_code(shape : @plan.OperationShape) -> Int {
  match shape {
    EmbeddingShape(..) => 1
    NormalizationShape(..) => 2
    ProjectionShape(..) => 3
    RotaryShape(..) => 4
    AttentionShape(..) => 5
    ElementwiseShape(..) => 6
    GatedMlpShape(..) => 7
  }
}

///|
fn LunaSpecializationEncoder::append_operation_shape(
  self : LunaSpecializationEncoder,
  shape : @plan.OperationShape,
) -> Unit raise LunaSpecializationError {
  self.append_int(operation_shape_code(shape))
  match shape {
    EmbeddingShape(vocabulary_size~, width~) => {
      self.append_int(vocabulary_size)
      self.append_int(width)
    }
    NormalizationShape(width~, epsilon~) => {
      self.append_int(width)
      self.append_u64(epsilon.reinterpret_as_uint64())
    }
    ProjectionShape(input_width~, output_width~) => {
      self.append_int(input_width)
      self.append_int(output_width)
    }
    RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~) => {
      self.append_int(query_heads)
      self.append_int(kv_heads)
      self.append_int(head_dimension)
      self.append_u64(rope_theta.reinterpret_as_uint64())
    }
    AttentionShape(query_heads~, kv_heads~, head_dimension~) => {
      self.append_int(query_heads)
      self.append_int(kv_heads)
      self.append_int(head_dimension)
    }
    ElementwiseShape(width~) => self.append_int(width)
    GatedMlpShape(hidden_size~, intermediate_size~) => {
      self.append_int(hidden_size)
      self.append_int(intermediate_size)
    }
  }
}

///|
fn LunaSpecializationEncoder::append_compiler(
  self : LunaSpecializationEncoder,
  policy : LunaCompilerPolicy,
) -> Unit raise LunaSpecializationError {
  self.append_string(policy.identity.toolchain_digest)
  self.append_int(policy.identity.major)
  self.append_int(policy.identity.minor)
  self.append_int(policy.identity.patch)
  self.append_int(policy.optimization_level)
  self.append_int(if policy.fused_multiply_add { 1 } else { 0 })
  self.append_int(if policy.reassociate { 1 } else { 0 })
  self.append_int(policy.max_registers)
}

///|
fn LunaSpecializationEncoder::append_numerical(
  self : LunaSpecializationEncoder,
  policy : LunaNumericalPolicy,
) -> Unit raise LunaSpecializationError {
  match policy {
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
fn encode_luna_specialization(
  identity : @spec.ModelIdentity,
  target : @catalog.DeviceTarget,
  catalog_version : @catalog.CatalogVersion,
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  module_digest : @catalog.AotArtifactDigest,
  entry_point_id : LunaSpecializedEntryPointId,
  profile : @device_profile.DeviceExecutionProfile,
  compiler : LunaCompilerPolicy,
  numerical : LunaNumericalPolicy,
  regions : ReadOnlyArray[LunaSpecializedRegion],
  rope_theta_bits : UInt64,
  rms_epsilon_bits : UInt64,
  program : @luna_tile_ir.LunaTileProgram,
  cuda_aot : @luna_tile_ir.LunaTileCudaAotInput,
  evidence : LunaSpecializationEvidence,
  maximum : Int,
) -> Bytes raise LunaSpecializationError {
  let encoder : LunaSpecializationEncoder = { bytes: [], maximum }
  encoder.append_string("LUNA-SPECIALIZATION-V1")
  encoder.append_string(identity.content().as_hex())
  encoder.append_string(identity.plan().as_hex())
  encoder.append_int(target.compute_major())
  encoder.append_int(target.compute_minor())
  encoder.append_int(if target.supports_bf16() { 1 } else { 0 })
  encoder.append_int(if target.supports_cublas_lt() { 1 } else { 0 })
  encoder.append_int(catalog_version.as_int())
  encoder.append_int(operation.id().as_int())
  encoder.append_int(operation.required_capability().as_int())
  encoder.append_int(binding.semantic_version().as_int())
  encoder.append_string(module_digest.sha256())
  encoder.append_int(entry_point_id.as_int())
  encoder.append_operation_shape(operation.shape())
  let shape = profile.shape()
  encoder.append_int(shape.phase().as_stable_int())
  encoder.append_int(shape.batch_rows())
  encoder.append_int(shape.query_tokens_per_row())
  encoder.append_int(shape.context_tokens())
  encoder.append_i64(shape.token_rows())
  encoder.append_i64(binding.workspace_bytes())
  encoder.append_i64(binding.workspace_alignment())
  encoder.append_u64(rope_theta_bits)
  encoder.append_u64(rms_epsilon_bits)
  encoder.append_int(1)
  encoder.append_compiler(compiler)
  encoder.append_numerical(numerical)
  encoder.append_string(program.digest().as_hex())
  encoder.append_string(cuda_aot.digest().as_hex())
  encoder.append_int(regions.length())
  for region in regions {
    encoder.append_int(
      match region.space {
        LunaWeightArena => 1
        LunaTokenInput => 2
        LunaActivationWorkspaceArena => 3
      },
    )
    encoder.append_int(region.semantic_id)
    encoder.append_i64(region.offset_bytes)
    encoder.append_i64(region.byte_length)
    encoder.append_i64(region.alignment_bytes)
  }
  for
    digest in [
      evidence.scalar_referee,
      evidence.admitted_tensor_rows,
      evidence.dispatch_canary,
      evidence.benchmark,
    ] {
    encoder.append_string(digest.as_hex())
  }
  Bytes::from_array(encoder.bytes)
}
