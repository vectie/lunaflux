///|
test "tensor-core qualification fixture is deterministic and authority-free" {
  let (serial, candidate) = tensor_core_probe_candidates() catch {
    LunaTensorCoreProbeRejected(reason) => fail(reason)
    _ => fail("unexpected tensor-core fixture failure")
  }
  let (serial_again, candidate_again) = tensor_core_probe_candidates()
  assert_eq(serial.digest(), serial_again.digest())
  assert_eq(candidate.digest(), candidate_again.digest())
  assert_eq(
    candidate.entry_point(),
    "lunaflux_lunatile_wmma_bf16_m16n16k16_sm120_v1",
  )
  assert_eq(candidate.tile_count(), 64)
  assert_false(candidate.manifest_bindable())
  assert_false(candidate.promotion_authorized())
  let record = @utf8.decode(tensor_core_probe_record(serial, candidate))
  for
    required in [
      "schema=lunaflux-lunatile-tensor-core-sm120-fixture.v1", "target=sm_120", "cpu_oracle=independent-ordered-f32-v1",
      "serial_cuda_oracle=required", "maximum_registers_per_thread=128", "maximum_static_shared_bytes=0",
      "qualification_only=true", "manifest_bindable=false", "promotion_authority=absent",
    ] {
    assert_true(record.contains(required))
  }
}

///|
test "hostile BF16 fixture and independent CPU oracle are stable" {
  let left = tensor_core_hostile_bf16(0, false)
  let right = tensor_core_hostile_bf16(0, true)
  assert_eq(left.length(), 32768)
  assert_eq(right.length(), 32768)
  assert_false(Bytes::from_array(left[:]) == Bytes::from_array(right[:]))
  let first = tensor_core_cpu_oracle(0)
  let repeated = tensor_core_cpu_oracle(0)
  let next = tensor_core_cpu_oracle(1)
  assert_eq(first, repeated)
  assert_false(first == next)
  let encoded = FixedArray::make(65536, b'\x00')
  for index, value in first {
    tensor_core_write_f32(encoded, index, value)
  }
  assert_eq(
    validate_tensor_core_output(
      Bytes::from_array(encoded[:]),
      first,
      "CPU self-check",
    ),
    0.0F,
  )
}
