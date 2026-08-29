///|
priv struct BootstrapDigestEncoder {
  bytes : Array[Byte]
}

///|
fn BootstrapDigestEncoder::append_u64(
  self : BootstrapDigestEncoder,
  value : UInt64,
) -> Unit {
  for byte in value.to_le_bytes() {
    self.bytes.push(byte)
  }
}

///|
fn BootstrapDigestEncoder::append_int(
  self : BootstrapDigestEncoder,
  value : Int,
) -> Unit {
  self.append_u64(value.to_int64().reinterpret_as_uint64())
}

///|
fn BootstrapDigestEncoder::append_int64(
  self : BootstrapDigestEncoder,
  value : Int64,
) -> Unit {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn BootstrapDigestEncoder::append_bool(
  self : BootstrapDigestEncoder,
  value : Bool,
) -> Unit {
  self.append_int(if value { 1 } else { 0 })
}

///|
fn BootstrapDigestEncoder::append_string(
  self : BootstrapDigestEncoder,
  value : StringView,
) -> Unit {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.bytes.push(byte)
  }
}

///|
fn BootstrapDigestEncoder::append_identity(
  self : BootstrapDigestEncoder,
  identity : @spec.ModelIdentity,
) -> Unit {
  self.append_string(identity.content().as_hex())
  self.append_string(identity.plan().as_hex())
}

///|
fn BootstrapDigestEncoder::append_target(
  self : BootstrapDigestEncoder,
  target : @catalog.DeviceTarget,
) -> Unit {
  self.append_int(target.compute_major())
  self.append_int(target.compute_minor())
  self.append_bool(target.supports_bf16())
  self.append_bool(target.supports_cublas_lt())
}

///|
fn BootstrapDigestEncoder::finish(self : BootstrapDigestEncoder) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(Bytes::from_array(self.bytes)))
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn group_digest_from_hex(
  value : String,
) -> TensorParallelGroupBootstrapDigest raise TensorParallelWorkerBootstrapError {
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  { hex: value }
}

///|
fn topology_digest_from_hex(
  value : String,
) -> TensorParallelTopologyDigest raise TensorParallelWorkerBootstrapError {
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  { hex: value }
}

///|
fn worker_digest_from_hex(
  value : String,
) -> TensorParallelWorkerContractDigest raise TensorParallelWorkerBootstrapError {
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  { hex: value }
}

///|
fn device_plan_digest_from_hex(
  value : String,
) -> TensorParallelDevicePlanDigest raise TensorParallelWorkerBootstrapError {
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  { hex: value }
}

///|
fn kv_binding_digest_from_hex(
  value : String,
) -> TensorParallelKvBindingDigest raise TensorParallelWorkerBootstrapError {
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  { hex: value }
}
