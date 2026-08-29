///|
fn bootstrap_i8_test_source() -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_i8_bootstrap_source_v6(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/model",
      kernel_root="/kernel",
    ),
    @worker_wire.WorkerI8BootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights.numeric",
      config_digest_sha256="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      numeric_weight_artifact_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="execution-v4.json",
      manifest_digest_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    ),
    @worker_wire.WorkerI8BootstrapExecution::new(
      compute_major=8,
      compute_minor=9,
      total_memory_bytes=8589934592L,
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
test "I8 source family is exact and legacy validation rejects it" {
  let source = bootstrap_i8_test_source()
  let startup = bootstrap_test_startup(source.digest())
  validate_i8_source_envelope(startup, source)
  assert_true(source.i8_model_source() is Some(_))
  assert_true(source.i8_execution() is Some(_))
  try validate_source_envelope(startup, source) catch {
    Invalid(Recipe) => ()
    _ => fail("unexpected legacy recipe failure")
  } noraise {
    _ => fail("legacy validation accepted I8 v6")
  }
}

///|
test "I8 source digest mismatch fails before root acquisition" {
  let source = bootstrap_i8_test_source()
  let startup = bootstrap_test_startup(
    @worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    ),
  )
  let acquisitions = Ref(0)
  try {
    validate_i8_source_envelope(startup, source)
    ignore(
      acquire_roots_with(_role => {
        acquisitions.val += 1
        raise Root(role=ModelRoot, issue=Unavailable)
      }),
    )
  } catch {
    Invalid(SourceDigest) => ()
    _ => fail("unexpected I8 source mismatch error")
  } noraise {
    _ => fail("I8 source mismatch was accepted")
  }
  assert_eq(acquisitions.val, 0)
}

///|
test "I8 limits bind exact graph and numeric tensor ceilings" {
  let source = bootstrap_i8_test_source()
  let startup = bootstrap_test_startup(source.digest())
  let (_, step, manifest, i8, bootstrap, numeric) = build_i8_limits(
    startup, source, 12, 24,
  )
  assert_eq(step.max_rows(), 1)
  assert_eq(manifest.max_operations(), 12)
  assert_eq(
    i8,
    @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits::new(
      max_operations=12,
      max_weights=24,
      max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
    ),
  )
  ignore(bootstrap)
  assert_eq(numeric.max_tensor_count(), 24)
}
