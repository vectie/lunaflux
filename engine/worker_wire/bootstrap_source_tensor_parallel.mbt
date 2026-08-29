///|
const BOOTSTRAP_SOURCE_TP_FIXED_HEADER_BYTES : Int = 208

///|
const BOOTSTRAP_SOURCE_TP_RANK_BYTES : Int = 48

///|
fn tensor_parallel_header_bytes(policy : TensorParallelBootstrapPolicy) -> Int {
  BOOTSTRAP_SOURCE_TP_FIXED_HEADER_BYTES +
  policy.rank_count() * BOOTSTRAP_SOURCE_TP_RANK_BYTES
}

///|
pub fn admit_tensor_parallel_bootstrap_source(
  roots : WorkerBootstrapRoots,
  model : WorkerBootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerBootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
  policy : TensorParallelBootstrapPolicy,
) -> EncodedBootstrapSource raise WorkerWireError {
  if kernel.luna_approval() is Some(_) {
    raise InvalidBootstrapSource(Recipe)
  }
  let lengths = bootstrap_source_lengths(roots, model, kernel)
  let header_bytes = tensor_parallel_header_bytes(policy)
  let mut length = header_bytes + BOOTSTRAP_SOURCE_DIGEST_BYTES
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
  write_u32(storage, 4, BOOTSTRAP_SOURCE_VERSION_V5)
  write_i32(storage, 8, BOOTSTRAP_SOURCE_RECIPE_V5)
  write_i32(storage, 12, length)
  write_u32(storage, 16, 0)
  for index, item in lengths {
    write_i32(storage, 20 + index * 4, item)
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
  write_i32(storage, 184, policy.rank_count())
  write_i32(storage, 188, policy.minimum_collective_runtime_version_code())
  write_i32(storage, 192, policy.maximum_collective_runtime_version_code())
  write_u32(storage, 196, 0)
  write_i32(storage, 200, policy.collective_exchange_timeout_millis())
  write_u32(storage, 204, 0)
  for rank in 0..<policy.rank_count() {
    guard policy.rank(rank) is Some(claim) else {
      raise InvalidBootstrapSource(CanonicalEncoding)
    }
    let offset = BOOTSTRAP_SOURCE_TP_FIXED_HEADER_BYTES +
      rank * BOOTSTRAP_SOURCE_TP_RANK_BYTES
    write_i32(storage, offset, claim.rank())
    write_i32(storage, offset + 4, claim.device_ordinal())
    write_u64(
      storage,
      offset + 8,
      claim.memory_ceiling_bytes().reinterpret_as_uint64(),
    )
    write_u64(
      storage,
      offset + 16,
      claim.weight_reserved_bytes().reinterpret_as_uint64(),
    )
    write_u64(
      storage,
      offset + 24,
      claim.activation_workspace_reserved_bytes().reinterpret_as_uint64(),
    )
    write_u64(
      storage,
      offset + 32,
      claim.kv_reserved_bytes().reinterpret_as_uint64(),
    )
    write_u64(
      storage,
      offset + 40,
      claim.collective_reserved_bytes().reinterpret_as_uint64(),
    )
  }
  let mut cursor = header_bytes
  cursor = write_bootstrap_source_string(storage, cursor, roots.model_root())
  cursor = write_bootstrap_source_string(
    storage,
    cursor,
    model.config_locator(),
  )
  cursor = write_bootstrap_source_string(
    storage,
    cursor,
    model.weights_locator(),
  )
  cursor = write_bootstrap_source_string(storage, cursor, roots.kernel_root())
  cursor = write_bootstrap_source_string(
    storage,
    cursor,
    kernel.manifest_locator(),
  )
  let digest = bootstrap_source_digest_of(storage, cursor)
  write_bootstrap_source_digest(storage, cursor, digest.as_hex())
  {
    storage,
    length,
    digest,
    roots,
    model,
    kernel,
    execution,
    ceilings,
    recipe: DenseLlamaTensorParallelPagedAotV5,
    tensor_parallel_policy: Some(policy),
    i8_model_source: None,
    i8_execution: None,
    mistral_model_source: None,
    fp8_model_source: None,
    fp8_execution: None,
  }
}

///|
fn decode_tensor_parallel_bootstrap_source(
  source : FixedArray[Byte],
  length : Int,
) -> EncodedBootstrapSource raise WorkerWireError {
  if read_u32(source, 4) == BOOTSTRAP_SOURCE_VERSION_V8 {
    return decode_tensor_parallel_approved_bootstrap_source(source, length)
  }
  let rank_count = read_i32(source, 184)
  let header_bytes = BOOTSTRAP_SOURCE_TP_FIXED_HEADER_BYTES +
    rank_count * BOOTSTRAP_SOURCE_TP_RANK_BYTES
  let lengths = FixedArray::make(5, 0)
  let mut payload_bytes = 0
  for index in 0..<5 {
    let item = read_i32(source, 20 + index * 4)
    if item <= 0 || item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES {
      raise InvalidBootstrapSource(Length)
    }
    lengths[index] = item
    payload_bytes += item
  }
  if header_bytes + payload_bytes + BOOTSTRAP_SOURCE_DIGEST_BYTES != length {
    raise InvalidBootstrapSource(Length)
  }
  let values = FixedArray::make(5, "")
  let mut cursor = header_bytes
  for index in 0..<5 {
    values[index] = read_bootstrap_source_string(source, cursor, lengths[index])
    cursor += lengths[index]
  }
  let roots = WorkerBootstrapRoots::new(
    model_root=values[0],
    kernel_root=values[3],
  )
  let model = WorkerBootstrapModelSource::new(
    config_locator=values[1],
    weights_locator=values[2],
    config_digest_sha256=bootstrap_source_raw_digest(source, 120).as_hex(),
  )
  let kernel = WorkerBootstrapKernelSource::new(
    manifest_locator=values[4],
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 152).as_hex(),
  )
  let flags = read_u32(source, 48)
  let execution = WorkerBootstrapExecution::new(
    compute_major=read_i32(source, 40),
    compute_minor=read_i32(source, 44),
    supports_bf16=(flags & 1) != 0,
    supports_cublas_lt=(flags & 2) != 0,
    tokens_per_page=read_i32(source, 52),
    total_page_count=read_i32(source, 56),
  )
  let ceilings = WorkerBootstrapCeilings::new(
    max_model_config_bytes=read_i32(source, 60),
    max_weight_file_bytes=bootstrap_source_read_i64(source, 64),
    max_weight_arena_bytes=bootstrap_source_read_i64(source, 72),
    max_activation_arena_bytes=bootstrap_source_read_i64(source, 80),
    max_kv_arena_bytes=bootstrap_source_read_i64(source, 88),
    max_execution_manifest_bytes=bootstrap_source_read_i64(source, 96),
    max_module_bytes=bootstrap_source_read_i64(source, 104),
    max_total_module_bytes=bootstrap_source_read_i64(source, 112),
  )
  let ranks : Array[TensorParallelRankRuntimeClaim] = []
  for rank in 0..<rank_count {
    let offset = BOOTSTRAP_SOURCE_TP_FIXED_HEADER_BYTES +
      rank * BOOTSTRAP_SOURCE_TP_RANK_BYTES
    if read_i32(source, offset) != rank {
      raise InvalidBootstrapSource(CanonicalEncoding)
    }
    ranks.push(
      TensorParallelRankRuntimeClaim::new(
        rank~,
        device_ordinal=read_i32(source, offset + 4),
        memory_ceiling_bytes=bootstrap_source_read_i64(source, offset + 8),
        weight_reserved_bytes=bootstrap_source_read_i64(source, offset + 16),
        activation_workspace_reserved_bytes=bootstrap_source_read_i64(
          source,
          offset + 24,
        ),
        kv_reserved_bytes=bootstrap_source_read_i64(source, offset + 32),
        collective_reserved_bytes=bootstrap_source_read_i64(source, offset + 40),
      ),
    )
  }
  let policy = TensorParallelBootstrapPolicy::new(
    ranks,
    minimum_collective_runtime_version_code=read_i32(source, 188),
    maximum_collective_runtime_version_code=read_i32(source, 192),
    collective_exchange_timeout_millis=read_i32(source, 200),
  )
  let canonical = admit_tensor_parallel_bootstrap_source(
    roots, model, kernel, execution, ceilings, policy,
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
