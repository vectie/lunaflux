///|
fn bootstrap_qwen3_test_source(
  artifact? : String = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  manifest? : String = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_qwen3_bf16_paged_bootstrap_source_v12(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/model",
      kernel_root="/kernel",
    ),
    @worker_wire.WorkerQwen3BootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights.numeric",
      config_digest_sha256="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      numeric_weight_artifact_sha256=artifact,
      weight_manifest_sha256=manifest,
      tied_embeddings=true,
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="execution.json",
      manifest_digest_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=9,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=4,
      total_page_count=8,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=4096,
      max_weight_file_bytes=65536L,
      max_weight_arena_bytes=65536L,
      max_activation_arena_bytes=65536L,
      max_kv_arena_bytes=65536L,
      max_execution_manifest_bytes=65536L,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
    ),
  )
}

///|
test "Qwen3 worker source is exact and legacy validation rejects it" {
  let source = bootstrap_qwen3_test_source()
  let startup = bootstrap_test_startup(source.digest())
  let model = validate_qwen3_source_envelope(startup, source)
  assert_true(model.tied_embeddings())
  assert_true(source.mistral_model_source() is None)
  try validate_source_envelope(startup, source) catch {
    Invalid(Recipe) => ()
    _ => fail("Qwen3 source reported an unrelated legacy failure")
  } noraise {
    _ => fail("legacy worker validation accepted Qwen3 v12")
  }
}

///|
test "Qwen3 source substitution fails before root acquisition" {
  let admitted = bootstrap_qwen3_test_source()
  let startup = bootstrap_test_startup(admitted.digest())
  let foreign = bootstrap_qwen3_test_source(
    artifact="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  )
  let acquisitions = Ref(0)
  try {
    ignore(validate_qwen3_source_envelope(startup, foreign))
    ignore(
      acquire_roots_with(_role => {
        acquisitions.val += 1
        raise Root(role=ModelRoot, issue=Unavailable)
      }),
    )
  } catch {
    Invalid(SourceDigest) => ()
    _ => fail("Qwen3 substitution reported an unrelated failure")
  } noraise {
    _ => fail("foreign Qwen3 source was admitted")
  }
  assert_eq(acquisitions.val, 0)
}
