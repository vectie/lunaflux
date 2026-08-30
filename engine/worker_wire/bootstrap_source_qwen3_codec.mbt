///|
const QWEN3_BOOTSTRAP_SOURCE_VERSION_V12 : UInt = 12

///|
const QWEN3_BOOTSTRAP_SOURCE_RECIPE_V12 : Int = 12

///|
const QWEN3_BOOTSTRAP_SOURCE_HEADER_BYTES_V12 : Int = 264

///|
fn qwen3_source_lengths(
  roots : WorkerBootstrapRoots,
  model : WorkerQwen3BootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
) -> Array[Int] {
  [
    @utf8.encode(roots.model_root()).length(),
    @utf8.encode(model.config_locator()).length(),
    @utf8.encode(model.numeric_weights_locator()).length(),
    @utf8.encode(roots.kernel_root()).length(),
    @utf8.encode(kernel.manifest_locator()).length(),
  ]
}

///|
/// Canonically frames one dense-Qwen3 BF16 paged source. Recipe v12 rejects
/// fused, approval, untied-output, and non-BF16 modes; its reserved sampling
/// word distinguishes host sampling from authenticated embedded CUDA greedy.
pub fn admit_qwen3_bf16_paged_bootstrap_source_v12(
  roots : WorkerBootstrapRoots,
  model : WorkerQwen3BootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerBootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
) -> EncodedBootstrapSource raise WorkerWireError {
  if kernel.luna_approval() is Some(_) ||
    kernel.fused_runtime_source() is Some(_) ||
    !execution.supports_bf16() ||
    !model.tied_embeddings() {
    raise InvalidBootstrapSource(Recipe)
  }
  let lengths = qwen3_source_lengths(roots, model, kernel)
  let mut length = QWEN3_BOOTSTRAP_SOURCE_HEADER_BYTES_V12 +
    BOOTSTRAP_SOURCE_DIGEST_BYTES
  for item in lengths {
    if item <= 0 || item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES {
      raise InvalidLimit(BootstrapSourceBytes)
    }
    length += item
  }
  if length > BOOTSTRAP_SOURCE_CAPACITY {
    raise InvalidLimit(BootstrapSourceBytes)
  }
  let storage = FixedArray::make(BOOTSTRAP_SOURCE_CAPACITY, b'\x00')
  write_u32(storage, 0, BOOTSTRAP_SOURCE_MAGIC)
  write_u32(storage, 4, QWEN3_BOOTSTRAP_SOURCE_VERSION_V12)
  write_i32(storage, 8, QWEN3_BOOTSTRAP_SOURCE_RECIPE_V12)
  write_i32(storage, 12, length)
  write_u32(storage, 16, 0U)
  for index in 0..<5 {
    write_i32(storage, 20 + index * 4, lengths[index])
  }
  write_i32(storage, 40, execution.compute_major())
  write_i32(storage, 44, execution.compute_minor())
  let flags : UInt = (if execution.supports_bf16() { 1 } else { 0 }) |
    (if execution.supports_cublas_lt() { 2 } else { 0 })
  write_u32(storage, 48, flags)
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
  write_bootstrap_source_digest(storage, 216, model.weight_manifest_digest())
  write_i32(storage, 248, 1)
  write_u32(
    storage,
    252,
    match kernel.sampling_runtime() {
      HostSamplingRuntime => 0U
      EmbeddedCudaGreedyV1 => 1U
    },
  )
  write_u64(storage, 256, 0UL)
  let values = [
    roots.model_root(),
    model.config_locator(),
    model.numeric_weights_locator(),
    roots.kernel_root(),
    kernel.manifest_locator(),
  ]
  let mut cursor = QWEN3_BOOTSTRAP_SOURCE_HEADER_BYTES_V12
  for value in values {
    cursor = write_bootstrap_source_string(storage, cursor, value)
  }
  let digest = bootstrap_source_digest_of(storage, cursor)
  write_bootstrap_source_digest(storage, cursor, digest.as_hex())
  {
    storage,
    length,
    digest,
    roots,
    model: model.model,
    kernel,
    execution,
    ceilings,
    recipe: DenseQwen3Bf16PagedAotV12,
    tensor_parallel_policy: None,
    i8_model_source: None,
    i8_execution: None,
    mistral_model_source: None,
    qwen3_model_source: Some(model),
    fp8_model_source: None,
    fp8_execution: None,
  }
}

///|
fn decode_qwen3_bf16_paged_bootstrap_source_v12(
  source : FixedArray[Byte],
  length : Int,
) -> EncodedBootstrapSource raise WorkerWireError {
  let header = QWEN3_BOOTSTRAP_SOURCE_HEADER_BYTES_V12
  if source.length() < header ||
    length < header + BOOTSTRAP_SOURCE_DIGEST_BYTES ||
    length > BOOTSTRAP_SOURCE_CAPACITY ||
    read_u32(source, 0) != BOOTSTRAP_SOURCE_MAGIC ||
    read_u32(source, 4) != QWEN3_BOOTSTRAP_SOURCE_VERSION_V12 ||
    read_i32(source, 8) != QWEN3_BOOTSTRAP_SOURCE_RECIPE_V12 ||
    read_i32(source, 12) != length ||
    read_u32(source, 16) != 0U ||
    read_i32(source, 248) != 1 ||
    read_u32(source, 252) > 1U ||
    read_u64(source, 256) != 0UL {
    raise InvalidBootstrapSource(Reserved)
  }
  let lengths = FixedArray::make(5, 0)
  let mut payload = 0
  for index in 0..<5 {
    let item = read_i32(source, 20 + index * 4)
    if item <= 0 ||
      item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
      payload > BOOTSTRAP_SOURCE_CAPACITY - item {
      raise InvalidBootstrapSource(Length)
    }
    lengths[index] = item
    payload += item
  }
  if header + payload + BOOTSTRAP_SOURCE_DIGEST_BYTES != length {
    raise InvalidBootstrapSource(Length)
  }
  let values = FixedArray::make(5, "")
  let mut cursor = header
  for index in 0..<5 {
    values[index] = read_bootstrap_source_string(source, cursor, lengths[index])
    cursor += lengths[index]
  }
  let roots = WorkerBootstrapRoots::new(
    model_root=values[0],
    kernel_root=values[3],
  )
  let model = WorkerQwen3BootstrapModelSource::new(
    config_locator=values[1],
    numeric_weights_locator=values[2],
    config_digest_sha256=bootstrap_source_raw_digest(source, 120).as_hex(),
    numeric_weight_artifact_sha256=bootstrap_source_raw_digest(source, 184).as_hex(),
    weight_manifest_sha256=bootstrap_source_raw_digest(source, 216).as_hex(),
    tied_embeddings=true,
  )
  let kernel = WorkerBootstrapKernelSource::new(
    manifest_locator=values[4],
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 152).as_hex(),
    sampling_runtime=if read_u32(source, 252) == 1U {
      EmbeddedCudaGreedyV1
    } else {
      HostSamplingRuntime
    },
  )
  let execution = WorkerBootstrapExecution::new(
    compute_major=read_i32(source, 40),
    compute_minor=read_i32(source, 44),
    supports_bf16=(read_u32(source, 48) & 1) != 0,
    supports_cublas_lt=(read_u32(source, 48) & 2) != 0,
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
  let canonical = admit_qwen3_bf16_paged_bootstrap_source_v12(
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
