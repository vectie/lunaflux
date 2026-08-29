///|
fn mistral_source_fixture() -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_mistral_bootstrap_source_v7(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/mistral",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerMistralBootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights/model.numeric.safetensors",
      config_digest_sha256=source_digest_a(),
      numeric_weight_artifact_sha256=source_digest_b(),
      weight_manifest_sha256=source_digest_a(),
      sliding_window_tokens=4096,
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/mistral-paged.json",
      manifest_digest_sha256=source_digest_b(),
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
test "Mistral v7 full frame roundtrips exact authenticated claims" {
  let encoded = mistral_source_fixture()
  assert_eq(encoded.recipe(), DenseMistralBf16PagedAotV7)
  let frame = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(frame, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(frame, encoded.length())
  assert_eq(decoded.digest(), encoded.digest())
  guard decoded.mistral_model_source() is Some(model) else {
    fail("Mistral source claims were discarded")
  }
  assert_eq(model.config_digest(), source_digest_a())
  assert_eq(model.numeric_weight_artifact_digest(), source_digest_b())
  assert_eq(model.weight_manifest_digest(), source_digest_a())
  assert_eq(model.sliding_window_tokens(), 4096)
  assert_eq(decoded.kernel().manifest_digest(), source_digest_b())
  // Decoding owns a canonical detached frame.
  frame[120] = frame[120] ^ b'\x01'
  assert_eq(decoded.digest(), encoded.digest())
}

///|
test "Mistral v7 rejects malformed reserved integrity and framing bytes" {
  let encoded = mistral_source_fixture()
  for mutation in [16, 252, 256, 120] {
    let frame = FixedArray::make(encoded.capacity(), b'\x00')
    encoded.copy_to(frame, destination_offset=0)
    frame[mutation] = frame[mutation] ^ b'\x01'
    try @worker_wire.decode_bootstrap_source(frame, encoded.length()) catch {
      InvalidBootstrapSource(_) => ()
      _ => fail("malformed Mistral frame reported an unrelated failure")
    } noraise {
      _ => fail("malformed Mistral frame was admitted")
    }
  }
  let trailing = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(trailing, destination_offset=0)
  try
    @worker_wire.decode_bootstrap_source(trailing, encoded.length() + 1)
  catch {
    InvalidBootstrapSource(Length) => ()
    _ => fail("trailing Mistral byte reported an unrelated failure")
  } noraise {
    _ => fail("trailing Mistral byte was admitted")
  }
}

///|
test "Mistral source rejects incomplete family policy before framing" {
  try
    @worker_wire.WorkerMistralBootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights.numeric",
      config_digest_sha256=source_digest_a(),
      numeric_weight_artifact_sha256=source_digest_b(),
      weight_manifest_sha256=source_digest_a(),
      sliding_window_tokens=-1,
    )
  catch {
    InvalidBootstrapSource(_) => ()
    _ => fail("negative sliding-window reported an unrelated failure")
  } noraise {
    _ => fail("negative sliding-window entered a Mistral frame")
  }
}
