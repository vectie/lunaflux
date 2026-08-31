///|
fn i8_source_fixture() -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_i8_bootstrap_source_v6(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/models",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerI8BootstrapModelSource::new(
      config_locator="config.json",
      numeric_weights_locator="weights/model.numeric.safetensors",
      config_digest_sha256=source_digest_a(),
      numeric_weight_artifact_sha256=source_digest_b(),
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/paged-i8-v4.json",
      manifest_digest_sha256=source_digest_a(),
    ),
    @worker_wire.WorkerI8BootstrapExecution::new(
      compute_major=8,
      compute_minor=9,
      total_memory_bytes=25769803776L,
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
test "I8 v6 bootstrap source roundtrips exact artifact and device facts" {
  let encoded = i8_source_fixture()
  assert_eq(encoded.recipe(), DenseLlamaI8PagedAotV6)
  assert_eq(encoded.capacity(), @worker_wire.i8_bootstrap_source_v6_capacity())
  let transport = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let decoded = @worker_wire.decode_i8_bootstrap_source_v6(
    transport,
    encoded.length(),
  )
  assert_eq(decoded.digest(), encoded.digest())
  guard decoded.i8_model_source() is Some(model) else {
    fail("I8 model evidence was dropped")
  }
  guard decoded.i8_execution() is Some(execution) else {
    fail("I8 execution evidence was dropped")
  }
  assert_eq(model.config_locator(), "config.json")
  assert_eq(
    model.numeric_weights_locator(),
    "weights/model.numeric.safetensors",
  )
  assert_eq(model.config_digest(), source_digest_a())
  assert_eq(model.numeric_weight_artifact_digest(), source_digest_b())
  assert_eq(decoded.kernel().manifest_digest(), source_digest_a())
  assert_eq(execution.compute_major(), 8)
  assert_eq(execution.compute_minor(), 9)
  assert_eq(execution.total_memory_bytes(), 25769803776L)
  assert_true(execution.supports_bf16())
  assert_true(execution.supports_cublas_lt())
}

///|
test "common codec transports I8 v6 while I8 decoder rejects legacy" {
  let i8 = i8_source_fixture()
  let i8_bytes = FixedArray::make(i8.capacity(), b'\x00')
  i8.copy_to(i8_bytes, destination_offset=0)
  let common = @worker_wire.decode_bootstrap_source(i8_bytes, i8.length())
  assert_eq(common.recipe(), DenseLlamaI8PagedAotV6)

  let legacy = source_fixture()
  let legacy_bytes = FixedArray::make(legacy.capacity(), b'\x00')
  legacy.copy_to(legacy_bytes, destination_offset=0)
  try
    @worker_wire.decode_i8_bootstrap_source_v6(legacy_bytes, legacy.length())
  catch {
    InvalidBootstrapSource(Version) => ()
    _ => fail("I8 decoder reported the wrong legacy recipe failure")
  } noraise {
    _ => fail("I8 decoder accepted legacy source")
  }
  assert_true(legacy.i8_model_source() is None)
  assert_true(legacy.i8_execution() is None)

  let tensor_parallel = source_tp_fixture()
  let tp_bytes = FixedArray::make(tensor_parallel.capacity(), b'\x00')
  tensor_parallel.copy_to(tp_bytes, destination_offset=0)
  try
    @worker_wire.decode_i8_bootstrap_source_v6(
      tp_bytes,
      tensor_parallel.length(),
    )
  catch {
    InvalidBootstrapSource(Version) => ()
    _ => fail("I8 decoder reported the wrong tensor-parallel failure")
  } noraise {
    _ => fail("I8 decoder accepted tensor-parallel source")
  }
}

///|
test "common codec rejects foreign version paired with I8 v6 recipe" {
  let i8 = i8_source_fixture()
  let bytes = FixedArray::make(i8.capacity(), b'\x00')
  i8.copy_to(bytes, destination_offset=0)
  // Recipe remains v6, but a foreign version must fail at the common version
  // gate before any I8 payload interpretation.
  bytes[4] = b'\x0e'
  try @worker_wire.decode_bootstrap_source(bytes, i8.length()) catch {
    InvalidBootstrapSource(Version) => ()
    _ => fail("foreign version reported the wrong common-codec failure")
  } noraise {
    _ => fail("foreign version entered I8 recipe decoding")
  }
}

///|
test "I8 v6 source rejects unsupported target and legacy approval" {
  let blackwell = @worker_wire.WorkerI8BootstrapExecution::new(
    compute_major=12,
    compute_minor=0,
    total_memory_bytes=25769803776L,
    supports_bf16=true,
    supports_cublas_lt=true,
    tokens_per_page=16,
    total_page_count=4096,
  )
  assert_eq(blackwell.compute_major(), 12)
  assert_eq(blackwell.compute_minor(), 0)
  assert_true(blackwell.supports_bf16())
  assert_true(blackwell.supports_cublas_lt())
  let blackwell_base = i8_source_fixture()
  guard blackwell_base.i8_model_source() is Some(blackwell_model) else {
    fail("I8 fixture lost typed model evidence")
  }
  let blackwell_source = @worker_wire.admit_i8_bootstrap_source_v6(
    blackwell_base.roots(),
    blackwell_model,
    blackwell_base.kernel(),
    blackwell,
    blackwell_base.ceilings(),
  )
  let blackwell_bytes = FixedArray::make(blackwell_source.capacity(), b'\x00')
  blackwell_source.copy_to(blackwell_bytes, destination_offset=0)
  let blackwell_decoded = @worker_wire.decode_i8_bootstrap_source_v6(
    blackwell_bytes,
    blackwell_source.length(),
  )
  guard blackwell_decoded.i8_execution() is Some(decoded_execution) else {
    fail("sm120 I8 execution facts were dropped by the wire codec")
  }
  assert_eq(decoded_execution.compute_major(), 12)
  assert_eq(decoded_execution.compute_minor(), 0)
  try
    @worker_wire.WorkerI8BootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      total_memory_bytes=25769803776L,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=16,
      total_page_count=4096,
    )
  catch {
    InvalidBootstrapSource(Target) => ()
    _ => fail("unexpected unsupported I8 target failure")
  } noraise {
    _ => fail("unsupported I8 target entered v6 source")
  }
  let base = i8_source_fixture()
  let approved_kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator=base.kernel().manifest_locator(),
    manifest_digest_sha256=base.kernel().manifest_digest(),
    luna_approval=Some(
      @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
        manifest_digest_sha256=source_digest_a(),
        key_identity_sha256=source_digest_b(),
        signature_digest_sha256=source_digest_a(),
        approved_source_sha256=source_digest_b(),
      ),
    ),
  )
  guard base.i8_model_source() is Some(base_model) &&
    base.i8_execution() is Some(base_execution) else {
    fail("I8 fixture lost typed evidence")
  }
  try
    @worker_wire.admit_i8_bootstrap_source_v6(
      base.roots(),
      base_model,
      approved_kernel,
      base_execution,
      base.ceilings(),
    )
  catch {
    InvalidBootstrapSource(Recipe) => ()
    _ => fail("unexpected legacy approval failure")
  } noraise {
    _ => fail("legacy approval entered production I8 source")
  }
}

///|
test "shared worker source capacity covers the maximum I8 v6 frame" {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root="/" + "a".repeat(4095),
    kernel_root="/" + "b".repeat(4095),
  )
  let model = @worker_wire.WorkerI8BootstrapModelSource::new(
    config_locator="c".repeat(4096),
    numeric_weights_locator="d".repeat(4096),
    config_digest_sha256=source_digest_a(),
    numeric_weight_artifact_sha256=source_digest_b(),
  )
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator="e".repeat(4096),
    manifest_digest_sha256=source_digest_a(),
  )
  let base = i8_source_fixture()
  guard base.i8_execution() is Some(base_execution) else {
    fail("I8 fixture lost execution evidence")
  }
  let maximum = @worker_wire.admit_i8_bootstrap_source_v6(
    roots,
    model,
    kernel,
    base_execution,
    base.ceilings(),
  )
  assert_eq(maximum.length(), @worker_wire.bootstrap_source_capacity())
  assert_eq(maximum.capacity(), @worker_wire.bootstrap_source_capacity())
}
