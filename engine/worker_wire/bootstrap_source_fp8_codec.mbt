///|
const FP8_BOOTSTRAP_SOURCE_VERSION_V9 : UInt = 9

///|
const FP8_BOOTSTRAP_SOURCE_RECIPE_V9 : Int = 9

///|
const FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9 : Int = 392

///|
pub fn admit_fp8_reusable_bootstrap_source_v9(
  roots : WorkerBootstrapRoots,
  model : WorkerFp8BootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerFp8BootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
) -> EncodedBootstrapSource raise WorkerWireError {
  guard kernel.luna_approval() is Some(approval) else {
    raise InvalidBootstrapSource(Recipe)
  }
  let lengths = FixedArray::from_array([
    @utf8.encode(roots.model_root()).length(),
    @utf8.encode(model.config_locator()).length(),
    @utf8.encode(model.numeric_weights_locator()).length(),
    @utf8.encode(roots.kernel_root()).length(),
    @utf8.encode(kernel.manifest_locator()).length(),
  ])
  let mut length = FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9 +
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
  write_u32(storage, 4, FP8_BOOTSTRAP_SOURCE_VERSION_V9)
  write_i32(storage, 8, FP8_BOOTSTRAP_SOURCE_RECIPE_V9)
  write_i32(storage, 12, length)
  write_u32(storage, 16, 1)
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
  write_bootstrap_source_digest(
    storage,
    216,
    execution.reusable_recipe_digest(),
  )
  write_bootstrap_source_digest(storage, 248, approval.manifest_digest())
  write_bootstrap_source_digest(storage, 280, approval.key_identity())
  write_bootstrap_source_digest(storage, 312, approval.signature_digest())
  write_bootstrap_source_digest(storage, 344, approval.approved_source())
  write_u64(
    storage,
    376,
    execution.total_memory_bytes().reinterpret_as_uint64(),
  )
  write_u64(storage, 384, 0UL)
  let mut cursor = FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9
  for
    value in [
      roots.model_root(),
      model.config_locator(),
      model.numeric_weights_locator(),
      roots.kernel_root(),
      kernel.manifest_locator(),
    ] {
    cursor = i8_source_write_string(storage, cursor, value)
  }
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
    recipe: DenseLlamaFp8ReusablePagedAotApprovedV9,
    tensor_parallel_policy: None,
    i8_model_source: None,
    i8_execution: None,
    mistral_model_source: None,
    fp8_model_source: Some(model),
    fp8_execution: Some(execution),
  }
}

///|
pub fn decode_fp8_reusable_bootstrap_source_v9(
  source : FixedArray[Byte],
  length : Int,
) -> EncodedBootstrapSource raise WorkerWireError {
  if source.length() < FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9 ||
    read_u32(source, 4) != FP8_BOOTSTRAP_SOURCE_VERSION_V9 ||
    read_i32(source, 8) != FP8_BOOTSTRAP_SOURCE_RECIPE_V9 ||
    read_u32(source, 16) != 1 ||
    read_u64(source, 384) != 0UL {
    raise InvalidBootstrapSource(Recipe)
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
  if FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9 +
    payload +
    BOOTSTRAP_SOURCE_DIGEST_BYTES !=
    length {
    raise InvalidBootstrapSource(Length)
  }
  let values = FixedArray::make(5, "")
  let mut cursor = FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9
  for index in 0..<5 {
    values[index] = i8_source_read_string(source, cursor, lengths[index])
    cursor += lengths[index]
  }
  let roots = WorkerBootstrapRoots::new(
    model_root=values[0],
    kernel_root=values[3],
  )
  let model = WorkerFp8BootstrapModelSource::new(
    values[1],
    values[2],
    bootstrap_source_raw_digest(source, 120).as_hex(),
    bootstrap_source_raw_digest(source, 184).as_hex(),
  )
  let approval = WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 248).as_hex(),
    key_identity_sha256=bootstrap_source_raw_digest(source, 280).as_hex(),
    signature_digest_sha256=bootstrap_source_raw_digest(source, 312).as_hex(),
    approved_source_sha256=bootstrap_source_raw_digest(source, 344).as_hex(),
  )
  let kernel = WorkerBootstrapKernelSource::new(
    manifest_locator=values[4],
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 152).as_hex(),
    luna_approval=Some(approval),
  )
  let execution = WorkerFp8BootstrapExecution::new(
    read_i32(source, 40),
    read_i32(source, 44),
    read_u64(source, 376).reinterpret_as_int64(),
    true,
    true,
    read_i32(source, 52),
    read_i32(source, 56),
    bootstrap_source_raw_digest(source, 216).as_hex(),
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
  let canonical = admit_fp8_reusable_bootstrap_source_v9(
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
