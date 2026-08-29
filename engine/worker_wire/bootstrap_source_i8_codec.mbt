///|
const I8_BOOTSTRAP_SOURCE_VERSION_V6 : UInt = 6

///|
const I8_BOOTSTRAP_SOURCE_RECIPE_V6 : Int = 6

///|
const I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6 : Int = 232

///|
const I8_BOOTSTRAP_SOURCE_CAPACITY_V6 : Int = BOOTSTRAP_SOURCE_CAPACITY

///|
pub fn i8_bootstrap_source_v6_capacity() -> Int {
  I8_BOOTSTRAP_SOURCE_CAPACITY_V6
}

///|
fn i8_source_lengths(
  roots : WorkerBootstrapRoots,
  model : WorkerI8BootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
) -> FixedArray[Int] {
  FixedArray::from_array([
    @utf8.encode(roots.model_root()).length(),
    @utf8.encode(model.config_locator()).length(),
    @utf8.encode(model.numeric_weights_locator()).length(),
    @utf8.encode(roots.kernel_root()).length(),
    @utf8.encode(kernel.manifest_locator()).length(),
  ])
}

///|
fn i8_source_digest_of(
  bytes : FixedArray[Byte],
  length : Int,
) -> WorkerBootstrapSourceDigest raise WorkerWireError {
  let snapshot = Bytes::makei(length, index => bytes[index])
  WorkerBootstrapSourceDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(snapshot)),
  )
}

///|
fn i8_source_write_string(
  destination : FixedArray[Byte],
  cursor : Int,
  value : StringView,
) -> Int {
  let encoded = @utf8.encode(value)
  for index in 0..<encoded.length() {
    destination[cursor + index] = encoded[index]
  }
  cursor + encoded.length()
}

///|
fn i8_source_read_string(
  source : FixedArray[Byte],
  cursor : Int,
  length : Int,
) -> String raise WorkerWireError {
  @utf8.decode(Bytes::makei(length, index => source[cursor + index])) catch {
    _ => raise InvalidBootstrapSource(Utf8)
  }
}

///|
pub fn admit_i8_bootstrap_source_v6(
  roots : WorkerBootstrapRoots,
  model : WorkerI8BootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerI8BootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
) -> EncodedBootstrapSource raise WorkerWireError {
  if kernel.luna_approval() is Some(_) {
    raise InvalidBootstrapSource(Recipe)
  }
  let lengths = i8_source_lengths(roots, model, kernel)
  let mut length = I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6 +
    BOOTSTRAP_SOURCE_DIGEST_BYTES
  for item in lengths {
    if item <= 0 || item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES {
      raise InvalidLimit(BootstrapSourceBytes)
    }
    length += item
  }
  if length > I8_BOOTSTRAP_SOURCE_CAPACITY_V6 {
    raise InvalidLimit(BootstrapSourceBytes)
  }
  let storage = FixedArray::make(I8_BOOTSTRAP_SOURCE_CAPACITY_V6, b'\x00')
  write_u32(storage, 0, BOOTSTRAP_SOURCE_MAGIC)
  write_u32(storage, 4, I8_BOOTSTRAP_SOURCE_VERSION_V6)
  write_i32(storage, 8, I8_BOOTSTRAP_SOURCE_RECIPE_V6)
  write_i32(storage, 12, length)
  write_u32(storage, 16, 0)
  for index, item in lengths {
    write_i32(storage, 20 + index * 4, item)
  }
  write_i32(storage, 40, execution.compute_major())
  write_i32(storage, 44, execution.compute_minor())
  write_u32(storage, 48, 3)
  write_i32(storage, 52, execution.tokens_per_page())
  write_i32(storage, 56, execution.total_page_count())
  write_i32(storage, 60, ceilings.max_model_config_bytes())
  write_u64(
    storage,
    64,
    ceilings.max_weight_file_bytes().reinterpret_as_uint64(),
  )
  write_u64(
    storage,
    72,
    ceilings.max_weight_arena_bytes().reinterpret_as_uint64(),
  )
  write_u64(
    storage,
    80,
    ceilings.max_activation_arena_bytes().reinterpret_as_uint64(),
  )
  write_u64(storage, 88, ceilings.max_kv_arena_bytes().reinterpret_as_uint64())
  write_u64(
    storage,
    96,
    ceilings.max_execution_manifest_bytes().reinterpret_as_uint64(),
  )
  write_u64(storage, 104, ceilings.max_module_bytes().reinterpret_as_uint64())
  write_u64(
    storage,
    112,
    ceilings.max_total_module_bytes().reinterpret_as_uint64(),
  )
  write_bootstrap_source_digest(storage, 120, model.config_digest())
  write_bootstrap_source_digest(storage, 152, kernel.manifest_digest())
  write_bootstrap_source_digest(
    storage,
    184,
    model.numeric_weight_artifact_digest(),
  )
  write_u64(
    storage,
    216,
    execution.total_memory_bytes().reinterpret_as_uint64(),
  )
  write_u64(storage, 224, 0UL)
  let mut cursor = I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6
  cursor = i8_source_write_string(storage, cursor, roots.model_root())
  cursor = i8_source_write_string(storage, cursor, model.config_locator())
  cursor = i8_source_write_string(
    storage,
    cursor,
    model.numeric_weights_locator(),
  )
  cursor = i8_source_write_string(storage, cursor, roots.kernel_root())
  cursor = i8_source_write_string(storage, cursor, kernel.manifest_locator())
  let digest = i8_source_digest_of(storage, cursor)
  write_bootstrap_source_digest(storage, cursor, digest.as_hex())
  {
    storage,
    length,
    digest,
    roots,
    model: model.model,
    kernel,
    execution: execution.execution,
    ceilings,
    recipe: DenseLlamaI8PagedAotV6,
    tensor_parallel_policy: None,
    i8_model_source: Some(model),
    i8_execution: Some(execution),
    mistral_model_source: None,
    qwen3_model_source: None,
    fp8_model_source: None,
    fp8_execution: None,
  }
}

///|
fn validate_i8_source_header(
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise WorkerWireError {
  if source.length() < I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6 ||
    length < I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6 + BOOTSTRAP_SOURCE_DIGEST_BYTES ||
    length > I8_BOOTSTRAP_SOURCE_CAPACITY_V6 ||
    source.length() < length ||
    read_i32(source, 12) != length {
    raise InvalidBootstrapSource(Length)
  }
  if read_u32(source, 0) != BOOTSTRAP_SOURCE_MAGIC {
    raise InvalidBootstrapSource(Magic)
  }
  if read_u32(source, 4) != I8_BOOTSTRAP_SOURCE_VERSION_V6 {
    raise InvalidBootstrapSource(Version)
  }
  if read_i32(source, 8) != I8_BOOTSTRAP_SOURCE_RECIPE_V6 {
    raise InvalidBootstrapSource(Recipe)
  }
  if read_u32(source, 16) != 0 ||
    read_u32(source, 48) != 3 ||
    read_u64(source, 224) != 0UL {
    raise InvalidBootstrapSource(Reserved)
  }
  let expected = i8_source_digest_of(
    source,
    length - BOOTSTRAP_SOURCE_DIGEST_BYTES,
  )
  if bootstrap_source_raw_digest(source, length - BOOTSTRAP_SOURCE_DIGEST_BYTES) !=
    expected {
    raise InvalidBootstrapSource(Integrity)
  }
}

///|
pub fn decode_i8_bootstrap_source_v6(
  source : FixedArray[Byte],
  length : Int,
) -> EncodedBootstrapSource raise WorkerWireError {
  validate_i8_source_header(source, length)
  let lengths = FixedArray::make(5, 0)
  let mut payload_length = 0
  for index in 0..<5 {
    let item = read_i32(source, 20 + index * 4)
    if item <= 0 ||
      item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
      payload_length > I8_BOOTSTRAP_SOURCE_CAPACITY_V6 - item {
      raise InvalidBootstrapSource(Length)
    }
    lengths[index] = item
    payload_length += item
  }
  if I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6 +
    payload_length +
    BOOTSTRAP_SOURCE_DIGEST_BYTES !=
    length {
    raise InvalidBootstrapSource(Length)
  }
  let values = FixedArray::make(5, "")
  let mut cursor = I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6
  for index in 0..<5 {
    values[index] = i8_source_read_string(source, cursor, lengths[index])
    cursor += lengths[index]
  }
  let roots = WorkerBootstrapRoots::new(
    model_root=values[0],
    kernel_root=values[3],
  )
  let model = WorkerI8BootstrapModelSource::new(
    config_locator=values[1],
    numeric_weights_locator=values[2],
    config_digest_sha256=bootstrap_source_raw_digest(source, 120).as_hex(),
    numeric_weight_artifact_sha256=bootstrap_source_raw_digest(source, 184).as_hex(),
  )
  let kernel = WorkerBootstrapKernelSource::new(
    manifest_locator=values[4],
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 152).as_hex(),
  )
  let execution = WorkerI8BootstrapExecution::new(
    compute_major=read_i32(source, 40),
    compute_minor=read_i32(source, 44),
    total_memory_bytes=read_u64(source, 216).reinterpret_as_int64(),
    supports_bf16=true,
    supports_cublas_lt=true,
    tokens_per_page=read_i32(source, 52),
    total_page_count=read_i32(source, 56),
  )
  let ceilings = WorkerBootstrapCeilings::new(
    max_model_config_bytes=read_i32(source, 60),
    max_weight_file_bytes=read_u64(source, 64).reinterpret_as_int64(),
    max_weight_arena_bytes=read_u64(source, 72).reinterpret_as_int64(),
    max_activation_arena_bytes=read_u64(source, 80).reinterpret_as_int64(),
    max_kv_arena_bytes=read_u64(source, 88).reinterpret_as_int64(),
    max_execution_manifest_bytes=read_u64(source, 96).reinterpret_as_int64(),
    max_module_bytes=read_u64(source, 104).reinterpret_as_int64(),
    max_total_module_bytes=read_u64(source, 112).reinterpret_as_int64(),
  )
  let canonical = admit_i8_bootstrap_source_v6(
    roots, model, kernel, execution, ceilings,
  )
  if canonical.length != length {
    raise InvalidBootstrapSource(CanonicalEncoding)
  }
  for index in 0..<length {
    if canonical.storage[index] != source[index] {
      raise InvalidBootstrapSource(CanonicalEncoding)
    }
  }
  canonical
}

///|
