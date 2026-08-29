///|
priv struct LunaKernelProfileEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn LunaKernelProfileEncoder::append_byte(
  self : LunaKernelProfileEncoder,
  value : Byte,
) -> Unit raise LunaKernelProfileError {
  if self.bytes.length() >= self.maximum {
    raise LunaKernelProfileInvalid(CanonicalBytes)
  }
  self.bytes.push(value)
}

///|
fn LunaKernelProfileEncoder::append_int(
  self : LunaKernelProfileEncoder,
  value : Int,
) -> Unit raise LunaKernelProfileError {
  for byte in Int64::from_int(value).reinterpret_as_uint64().to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn LunaKernelProfileEncoder::append_i64(
  self : LunaKernelProfileEncoder,
  value : Int64,
) -> Unit raise LunaKernelProfileError {
  for byte in value.reinterpret_as_uint64().to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn LunaKernelProfileEncoder::append_string(
  self : LunaKernelProfileEncoder,
  value : StringView,
) -> Unit raise LunaKernelProfileError {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
}

///|
fn encode_luna_kernel_profile(
  identity : @spec.ModelIdentity,
  target : @catalog.DeviceTarget,
  workload : LunaKernelProfileDigest,
  profiler : LunaKernelProfileDigest,
  sampled_wall_time_ns : Int64,
  attributed_self_time_ns : Int64,
  observations : ArrayView[LunaKernelProfileObservation],
  maximum : Int,
) -> Bytes raise LunaKernelProfileError {
  let encoder : LunaKernelProfileEncoder = { bytes: [], maximum }
  encoder.append_string("lunaflux.kernel-profile-priority.v1")
  encoder.append_string(identity.content().as_hex())
  encoder.append_string(identity.plan().as_hex())
  encoder.append_int(target.compute_major())
  encoder.append_int(target.compute_minor())
  encoder.append_int(if target.supports_bf16() { 1 } else { 0 })
  encoder.append_int(if target.supports_cublas_lt() { 1 } else { 0 })
  encoder.append_string(workload.as_hex())
  encoder.append_string(profiler.as_hex())
  encoder.append_i64(sampled_wall_time_ns)
  encoder.append_i64(attributed_self_time_ns)
  encoder.append_int(observations.length())
  for observation in observations {
    encoder.append_int(observation.operation_id().as_int())
    encoder.append_int(observation.shape().phase().as_stable_int())
    encoder.append_int(observation.shape().batch_rows())
    encoder.append_int(observation.shape().query_tokens_per_row())
    encoder.append_int(observation.shape().context_tokens())
    encoder.append_i64(observation.call_count())
    encoder.append_i64(observation.self_time_ns())
    encoder.append_i64(observation.inclusive_time_ns())
  }
  Bytes::from_array(encoder.bytes)
}
