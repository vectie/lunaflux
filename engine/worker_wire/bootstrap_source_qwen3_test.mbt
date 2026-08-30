///|
fn qwen3_source_fixture(
  sampling_runtime? : @worker_wire.WorkerBootstrapSamplingRuntime = HostSamplingRuntime,
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_qwen3_bf16_paged_bootstrap_source_v12(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/qwen3",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerQwen3BootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights/model.numeric",
      config_digest_sha256=source_digest_a(),
      numeric_weight_artifact_sha256=source_digest_b(),
      weight_manifest_sha256=source_digest_a(),
      tied_embeddings=true,
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/qwen3-bf16-paged.json",
      manifest_digest_sha256=source_digest_b(),
      sampling_runtime~,
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=9,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=16,
      total_page_count=4096,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1048576,
      max_weight_file_bytes=8589934592L,
      max_weight_arena_bytes=8589934592L,
      max_activation_arena_bytes=1073741824L,
      max_kv_arena_bytes=4294967296L,
      max_execution_manifest_bytes=1048576L,
      max_module_bytes=67108864L,
      max_total_module_bytes=268435456L,
    ),
  )
}

///|
test "Qwen3 v12 roundtrips exact tied BF16 source claims" {
  let encoded = qwen3_source_fixture()
  assert_eq(encoded.recipe(), DenseQwen3Bf16PagedAotV12)
  assert_true(encoded.mistral_model_source() is None)
  let frame = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(frame, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(frame, encoded.length())
  assert_eq(decoded.digest(), encoded.digest())
  guard decoded.qwen3_model_source() is Some(model) else {
    fail("Qwen3 source claims were discarded")
  }
  assert_true(model.tied_embeddings())
  assert_eq(model.numeric_weight_artifact_digest(), source_digest_b())
  assert_eq(model.weight_manifest_digest(), source_digest_a())
}

///|
test "Qwen3 v12 roundtrips embedded CUDA greedy sampling" {
  let encoded = qwen3_source_fixture(sampling_runtime=EmbeddedCudaGreedyV1)
  let frame = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(frame, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(frame, encoded.length())
  assert_eq(decoded.digest(), encoded.digest())
  assert_eq(decoded.kernel().sampling_runtime(), EmbeddedCudaGreedyV1)
}

///|
test "Qwen3 v12 rejects untied runtime modes" {
  try
    @worker_wire.WorkerQwen3BootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights.numeric",
      config_digest_sha256=source_digest_a(),
      numeric_weight_artifact_sha256=source_digest_b(),
      weight_manifest_sha256=source_digest_a(),
      tied_embeddings=false,
    )
  catch {
    InvalidBootstrapSource(Recipe) => ()
    _ => fail("untied Qwen3 reported an unrelated failure")
  } noraise {
    _ => fail("untied Qwen3 entered a bootstrap frame")
  }
}
