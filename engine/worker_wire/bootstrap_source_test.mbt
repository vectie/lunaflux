///|
fn source_digest_a() -> String {
  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}

///|
fn source_digest_b() -> String {
  "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
}

///|
fn source_fixture() -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/models",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights/model.safetensors",
      config_digest_sha256=source_digest_a(),
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/paged-v1.json",
      manifest_digest_sha256=source_digest_b(),
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
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
test "canonical bootstrap source has stable bytes, digest, and roundtrip" {
  let encoded = source_fixture()
  assert_eq(encoded.recipe(), DenseLlamaPagedAotV1)
  assert_eq(encoded.capacity(), @worker_wire.bootstrap_source_capacity())
  assert_eq(encoded.length(), 316)
  assert_eq(
    encoded.digest().as_hex(),
    "149fe5d3cc1b528ee15c570deac49581f15bab1411af3617850c3b01f3975acb",
  )
  let transport = FixedArray::make(encoded.capacity(), b'\x55')
  encoded.copy_to(transport, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(
    transport,
    encoded.length(),
  )
  assert_eq(decoded.digest(), encoded.digest())
  assert_eq(decoded.roots().model_root(), "/srv/lunaflux/models")
  assert_eq(decoded.roots().kernel_root(), "/srv/lunaflux/kernels")
  assert_eq(decoded.model().config_locator(), "config.json")
  assert_eq(decoded.model().weights_locator(), "weights/model.safetensors")
  assert_eq(decoded.model().config_digest(), source_digest_a())
  assert_eq(decoded.kernel().manifest_locator(), "manifests/paged-v1.json")
  assert_eq(decoded.kernel().manifest_digest(), source_digest_b())
  assert_eq(decoded.execution().compute_major(), 8)
  assert_eq(decoded.execution().compute_minor(), 0)
  assert_true(decoded.execution().supports_bf16())
  assert_false(decoded.execution().supports_cublas_lt())
  assert_eq(decoded.execution().tokens_per_page(), 16)
  assert_eq(decoded.execution().total_page_count(), 4096)
  assert_eq(decoded.ceilings().max_model_config_bytes(), 1048576)
  assert_eq(decoded.ceilings().max_weight_file_bytes(), 8589934592L)
  assert_eq(decoded.ceilings().max_weight_arena_bytes(), 8589934592L)
  assert_eq(decoded.ceilings().max_activation_arena_bytes(), 1073741824L)
  assert_eq(decoded.ceilings().max_kv_arena_bytes(), 4294967296L)
  assert_eq(decoded.ceilings().max_execution_manifest_bytes(), 1048576L)
  assert_eq(decoded.ceilings().max_module_bytes(), 67108864L)
  assert_eq(decoded.ceilings().max_total_module_bytes(), 268435456L)
  let second = FixedArray::make(decoded.capacity(), b'\x00')
  decoded.copy_to(second, destination_offset=0)
  for index in 0..<encoded.length() {
    assert_eq(second[index], transport[index])
  }
}

///|
test "bootstrap source copies only its exact bounded range" {
  let encoded = source_fixture()
  let destination = FixedArray::make(encoded.length() + 2, b'\x7f')
  encoded.copy_to(destination, destination_offset=1)
  assert_eq(destination[0], b'\x7f')
  assert_eq(destination[encoded.length() + 1], b'\x7f')
  try
    encoded.copy_to(FixedArray::make(8, b'\x00'), destination_offset=0)
  catch {
    DestinationTooSmall => ()
    _ => fail("unexpected bootstrap-source copy error")
  } noraise {
    _ => fail("bootstrap source copied into undersized storage")
  }
  try encoded.copy_to(destination, destination_offset=-1) catch {
    DestinationTooSmall => ()
    _ => fail("unexpected negative bootstrap-source copy error")
  } noraise {
    _ => fail("bootstrap source accepted a negative copy offset")
  }
}

///|
test "decoded bootstrap source owns bytes independently of transport" {
  let encoded = source_fixture()
  let transport = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(
    transport,
    encoded.length(),
  )
  transport[184] = b'X'
  transport[encoded.length() - 1] = b'X'
  let retained = FixedArray::make(decoded.capacity(), b'\x00')
  let original = FixedArray::make(encoded.capacity(), b'\x00')
  decoded.copy_to(retained, destination_offset=0)
  encoded.copy_to(original, destination_offset=0)
  for index in 0..<encoded.length() {
    assert_eq(retained[index], original[index])
  }
}

///|
test "bootstrap source admits its exact maximum canonical capacity" {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root="/" + "a".repeat(4095),
    kernel_root="/" + "b".repeat(4095),
  )
  let model = @worker_wire.WorkerBootstrapModelSource::new(
    config_locator="c".repeat(4096),
    weights_locator="d".repeat(4096),
    config_digest_sha256=source_digest_a(),
  )
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator="e".repeat(4096),
    manifest_digest_sha256=source_digest_b(),
  )
  let base = source_fixture()
  let maximum = @worker_wire.admit_bootstrap_source(
    roots,
    model,
    kernel,
    base.execution(),
    base.ceilings(),
  )
  assert_eq(maximum.length(), @worker_wire.bootstrap_source_capacity())
  let bytes = FixedArray::make(maximum.capacity(), b'\x00')
  maximum.copy_to(bytes, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(bytes, maximum.length())
  assert_eq(decoded.length(), maximum.length())
  assert_eq(decoded.digest(), maximum.digest())
}

///|
test "bootstrap source constructors reject ambient or unbounded paths" {
  try
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="relative/models",
      kernel_root="/srv/kernels",
    )
  catch {
    InvalidBootstrapSource(RootPath) => ()
    _ => fail("unexpected relative-root error")
  } noraise {
    _ => fail("relative root locator was accepted")
  }
  try
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="../config.json",
      weights_locator="weights/model.safetensors",
      config_digest_sha256=source_digest_a(),
    )
  catch {
    InvalidBootstrapSource(RelativeLocator) => ()
    _ => fail("unexpected traversal-locator error")
  } noraise {
    _ => fail("traversal locator was accepted")
  }
  try
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifest.json",
      manifest_digest_sha256="ABCDEF",
    )
  catch {
    InvalidBootstrapSource(Digest) => ()
    _ => fail("unexpected malformed-digest error")
  } noraise {
    _ => fail("malformed manifest digest was accepted")
  }
  for root in ["/srv//models", "/srv/models/", "/srv\\models"] {
    try
      @worker_wire.WorkerBootstrapRoots::new(
        model_root=root,
        kernel_root="/srv/kernels",
      )
    catch {
      InvalidBootstrapSource(RootPath) => ()
      _ => fail("unexpected malformed-root error")
    } noraise {
      _ => fail("malformed root locator was accepted")
    }
  }
  for locator in ["weights//model", "weights/model/", "weights\\model"] {
    try
      @worker_wire.WorkerBootstrapKernelSource::new(
        manifest_locator=locator,
        manifest_digest_sha256=source_digest_b(),
      )
    catch {
      InvalidBootstrapSource(RelativeLocator) => ()
      _ => fail("unexpected malformed-locator error")
    } noraise {
      _ => fail("malformed relative locator was accepted")
    }
  }
}

///|
test "bootstrap source constructors reject unsafe scalar ceilings" {
  try
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=0,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
      tokens_per_page=16,
      total_page_count=1,
    )
  catch {
    InvalidBootstrapSource(Target) => ()
    _ => fail("unexpected target error")
  } noraise {
    _ => fail("invalid target was accepted")
  }
  try
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1,
      max_weight_file_bytes=9L,
      max_weight_arena_bytes=1L,
      max_activation_arena_bytes=1L,
      max_kv_arena_bytes=1L,
      max_execution_manifest_bytes=1L,
      max_module_bytes=2L,
      max_total_module_bytes=1L,
    )
  catch {
    InvalidBootstrapSource(Ceiling) => ()
    _ => fail("unexpected source-ceiling error")
  } noraise {
    _ => fail("inverted module ceiling was accepted")
  }
}
