///|
fn wb_source() -> EncodedBootstrapSource raise {
  admit_bootstrap_source(
    WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/models",
      kernel_root="/srv/lunaflux/kernels",
    ),
    WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights/model.safetensors",
      config_digest_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/paged-v2.json",
      manifest_digest_sha256="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
    WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
      tokens_per_page=16,
      total_page_count=4096,
    ),
    WorkerBootstrapCeilings::new(
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
fn wb_source_bytes() -> (FixedArray[Byte], Int) raise {
  let source = wb_source()
  let bytes = FixedArray::make(source.capacity(), b'\x00')
  source.copy_to(bytes, destination_offset=0)
  (bytes, source.length())
}

///|
fn wb_recompute_integrity(bytes : FixedArray[Byte], length : Int) -> Unit raise {
  let digest = bootstrap_source_digest_of(
    bytes,
    length - BOOTSTRAP_SOURCE_DIGEST_BYTES,
  )
  write_bootstrap_source_digest(
    bytes,
    length - BOOTSTRAP_SOURCE_DIGEST_BYTES,
    digest.as_hex(),
  )
}

///|
fn expect_source_rule(
  bytes : FixedArray[Byte],
  length : Int,
  expected : WorkerBootstrapSourceRule,
) -> Unit raise {
  try decode_bootstrap_source(bytes, length) catch {
    InvalidBootstrapSource(actual) => assert_eq(actual, expected)
    _ => fail("unexpected corrupted bootstrap-source error")
  } noraise {
    _ => fail("corrupted bootstrap source was accepted")
  }
}

///|
test "bootstrap source integrity covers every payload byte" {
  let (bytes, length) = wb_source_bytes()
  bytes[BOOTSTRAP_SOURCE_HEADER_BYTES_V2] = b'X'
  expect_source_rule(bytes, length, Integrity)
}

///|
test "bootstrap source validates structural header fields distinctly" {
  let (magic, length) = wb_source_bytes()
  magic[0] = b'X'
  expect_source_rule(magic, length, Magic)
  let (version, _) = wb_source_bytes()
  write_u32(version, 4, 1)
  expect_source_rule(version, length, Version)
  let (recipe, _) = wb_source_bytes()
  write_i32(recipe, 8, 1)
  expect_source_rule(recipe, length, Recipe)
  let (reserved, _) = wb_source_bytes()
  write_u32(reserved, 16, 1)
  expect_source_rule(reserved, length, Reserved)
  let (flags, _) = wb_source_bytes()
  write_u32(flags, 48, 4)
  expect_source_rule(flags, length, Reserved)
  let (declared, _) = wb_source_bytes()
  write_i32(declared, 12, length - 1)
  expect_source_rule(declared, length, Length)
}

///|
test "integrity-valid source still validates UTF-8 paths and ceilings" {
  let (utf8, length) = wb_source_bytes()
  utf8[BOOTSTRAP_SOURCE_HEADER_BYTES_V2] = b'\xff'
  wb_recompute_integrity(utf8, length)
  expect_source_rule(utf8, length, Utf8)

  let (traversal, _) = wb_source_bytes()
  let config_offset = BOOTSTRAP_SOURCE_HEADER_BYTES_V2 + read_i32(traversal, 20)
  let replacement = @utf8.encode("../fig.json")
  assert_eq(replacement.length(), read_i32(traversal, 24))
  for index in 0..<replacement.length() {
    traversal[config_offset + index] = replacement[index]
  }
  wb_recompute_integrity(traversal, length)
  expect_source_rule(traversal, length, RelativeLocator)

  let (ceiling, _) = wb_source_bytes()
  write_u64(ceiling, 104, 0UL)
  wb_recompute_integrity(ceiling, length)
  expect_source_rule(ceiling, length, Ceiling)
}

///|
test "integrity-valid source still validates target and KV geometry" {
  let (target, length) = wb_source_bytes()
  write_i32(target, 40, 0)
  wb_recompute_integrity(target, length)
  expect_source_rule(target, length, Target)

  let (tokens, _) = wb_source_bytes()
  write_i32(tokens, 52, 0)
  wb_recompute_integrity(tokens, length)
  expect_source_rule(tokens, length, KvLayout)

  let (pages, _) = wb_source_bytes()
  write_i32(pages, 56, 0)
  wb_recompute_integrity(pages, length)
  expect_source_rule(pages, length, KvLayout)
}

///|
test "bootstrap source rejects non-exact input lengths and footer digest" {
  let (short, length) = wb_source_bytes()
  expect_source_rule(short, length - 1, Length)
  let (footer, _) = wb_source_bytes()
  footer[length - 1] = footer[length - 1] ^ b'\x01'
  expect_source_rule(footer, length, Integrity)
  let tiny = FixedArray::make(8, b'\x00')
  expect_source_rule(tiny, 8, Length)
}
