///|
test "GLM batched recipe declares GPU sampling and does not alias dense bootstrap" {
  let base = source_fixture()
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator="glm-service.json",
    manifest_digest_sha256=source_digest_b(),
    sampling_runtime=DeviceFullF32SamplingV1,
  )
  let source = @worker_wire.admit_glm_batched_bootstrap_source(
    base.roots(),
    base.model(),
    kernel,
    base.execution(),
    base.ceilings(),
  )
  assert_eq(source.recipe(), PackedGlmBatchedTensorV14)
  let bytes = FixedArray::make(source.capacity(), b'\x00')
  source.copy_to(bytes, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(bytes, source.length())
  assert_eq(decoded.recipe(), PackedGlmBatchedTensorV14)
  assert_eq(decoded.kernel().sampling_runtime(), DeviceFullF32SamplingV1)
  assert_true(decoded.model() == base.model())
  assert_eq(decoded.digest(), source.digest())
  try
    @worker_wire.admit_bootstrap_source(
      base.roots(),
      base.model(),
      kernel,
      base.execution(),
      base.ceilings(),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("silently converted full GPU sampling into host sampling")
  }
}
