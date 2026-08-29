///|
fn fp8_scale_cell_valid(bits : UInt) -> Bool {
  (bits & 0x80000000) == 0 &&
  (bits & 0x7fffffff) != 0 &&
  (bits & 0x7f800000) != 0x7f800000
}

///|
fn fp8_scale_read_u32(bytes : FixedArray[Byte], offset : Int) -> UInt {
  bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)
}

///|
fn fill_fp8_scale_sentinel_v3(bytes : FixedArray[Byte]) -> Unit {
  for offset in 0..<(bytes.length() / 4) {
    let base = offset * 4
    bytes[base] = b'\x00'
    bytes[base + 1] = b'\x00'
    bytes[base + 2] = b'\xc0'
    bytes[base + 3] = b'\x7f'
  }
}

///|
fn arm_fp8_scale_evidence_v3(
  executor : PagedGraphExecutor,
) -> Unit raise DeviceStepError {
  guard executor.fp8_scale_evidence is Some(evidence) else { return }
  guard executor.resources.activation_workspace is Some(allocation) &&
    allocation.open else {
    raise InvalidExecutor(ResourceState)
  }
  if evidence.workspace_bytes <= 0L ||
    evidence.workspace_bytes % 4L != 0L ||
    evidence.host.length().to_int64() != evidence.workspace_bytes {
    raise InvalidExecutor(Fp8ScaleEvidence)
  }
  // Arm every per-operation cell before enqueue. Because the reusable recipe
  // gives each FP8 launch a disjoint 4/8-byte region, a skipped or failed early
  // operation cannot be hidden by a later operation's successful publication.
  fill_fp8_scale_sentinel_v3(evidence.host)
  let descriptor = complete_descriptor(executor.resources)
  descriptor.context.copy_from_fixed_host(
    allocation.resource,
    evidence.host,
    0L,
    evidence.workspace_base,
    evidence.workspace_bytes,
  ) catch {
    error => raise ExecutorDevice(ActivationReadback, error)
  }
}

///|
fn validate_fp8_scale_evidence_v3(
  executor : PagedGraphExecutor,
) -> Unit raise DeviceStepError {
  guard executor.fp8_scale_evidence is Some(evidence) else { return }
  guard executor.resources.activation_workspace is Some(allocation) &&
    allocation.open else {
    raise InvalidExecutor(ResourceState)
  }
  if evidence.workspace_bytes <= 0L ||
    evidence.workspace_bytes % 4L != 0L ||
    evidence.host.length().to_int64() != evidence.workspace_bytes {
    raise InvalidExecutor(Fp8ScaleEvidence)
  }
  let descriptor = complete_descriptor(executor.resources)
  descriptor.context.copy_to_fixed_host(
    allocation.resource,
    evidence.host,
    evidence.workspace_base,
    0L,
    evidence.workspace_bytes,
  ) catch {
    error => raise ExecutorDevice(ActivationReadback, error)
  }
  for offset in 0..<(evidence.host.length() / 4) {
    let bits = fp8_scale_read_u32(evidence.host, offset * 4)
    if !fp8_scale_cell_valid(bits) {
      raise InvalidExecutor(Fp8ScaleEvidence)
    }
  }
}
