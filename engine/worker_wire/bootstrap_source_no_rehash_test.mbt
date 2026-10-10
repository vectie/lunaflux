///|
test "all bootstrap receive recipes consume footer labels without reconstruction hashing" {
  let fused = @worker_wire.WorkerBootstrapFusedRuntimeSource::new(
    locator="aggregates/fused-runtime.v1",
    digest_sha256=source_digest_a(),
  )
  let mistral = mistral_source_fixture()
  let mistral_extended = @worker_wire.admit_mistral_bootstrap_source_v7(
    mistral.roots(),
    mistral.mistral_model_source().unwrap(),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator=mistral.kernel().manifest_locator(),
      manifest_digest_sha256=mistral.kernel().manifest_digest(),
      fused_runtime=Some(fused),
    ),
    mistral.execution(),
    mistral.ceilings(),
  )
  for
    encoded in [
      source_fixture(),
      source_v3_fixture(),
      source_v10_fused_fixture(),
      i8_source_fixture(),
      fp8_v9_source_fixture(),
      qwen3_source_fixture(),
      qwen3_source_fixture(fused_runtime=Some(fused)),
      mistral,
      mistral_extended,
      source_tp_fixture(),
      source_tp_approved_fixture(),
    ] {
    let frame = FixedArray::make(encoded.capacity(), b'\x00')
    encoded.copy_to(frame, destination_offset=0)
    frame[encoded.length() - 1] = frame[encoded.length() - 1] ^ b'\x01'
    let decoded = @worker_wire.decode_bootstrap_source(frame, encoded.length())
    assert_eq(decoded.recipe(), encoded.recipe())
    assert_true(decoded.roots() == encoded.roots())
    assert_eq(decoded.execution(), encoded.execution())
    assert_eq(decoded.ceilings(), encoded.ceilings())
    assert_true(decoded.digest() != encoded.digest())
    // The new owner retains the received label, not a newly hashed encoding.
    let retained = FixedArray::make(decoded.capacity(), b'\x00')
    decoded.copy_to(retained, destination_offset=0)
    for index in 0..<encoded.length() {
      assert_eq(retained[index], frame[index])
    }
    frame[encoded.length() - 1] = b'\x00'
    let owned_again = FixedArray::make(decoded.capacity(), b'\x00')
    decoded.copy_to(owned_again, destination_offset=0)
    assert_eq(owned_again[encoded.length() - 1], retained[encoded.length() - 1])
  }
}
