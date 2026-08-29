///|
const MISTRAL_BOOTSTRAP_SOURCE_VERSION_V7 : UInt = 7

///|
const MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V7 : Int = 7

///|
const MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V7 : Int = 264

///|
const MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11 : UInt = 11

///|
const MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V11 : Int = 11

///|
const MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V11 : Int = 296

///|
fn mistral_source_lengths(
  roots : WorkerBootstrapRoots,
  model : WorkerMistralBootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
) -> Array[Int] {
  let lengths = [
    @utf8.encode(roots.model_root()).length(),
    @utf8.encode(model.config_locator()).length(),
    @utf8.encode(model.numeric_weights_locator()).length(),
    @utf8.encode(roots.kernel_root()).length(),
    @utf8.encode(kernel.manifest_locator()).length(),
  ]
  match kernel.fused_runtime_source() {
    Some(source) => lengths.push(@utf8.encode(source.locator()).length())
    None => ()
  }
  lengths
}

///|
pub fn admit_mistral_bootstrap_source_v7(
  roots : WorkerBootstrapRoots,
  model : WorkerMistralBootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerBootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
) -> EncodedBootstrapSource raise WorkerWireError {
  if kernel.luna_approval() is Some(_) || !execution.supports_bf16() {
    raise InvalidBootstrapSource(Recipe)
  }
  let lengths = mistral_source_lengths(roots, model, kernel)
  let extensions = kernel.fused_runtime_source() is Some(_) ||
    kernel.sampling_runtime() == EmbeddedCudaGreedyV1
  let header_bytes = if extensions {
    MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V11
  } else {
    MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V7
  }
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
  write_u32(
    storage,
    4,
    if extensions {
      MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11
    } else {
      MISTRAL_BOOTSTRAP_SOURCE_VERSION_V7
    },
  )
  write_i32(
    storage,
    8,
    if extensions {
      MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V11
    } else {
      MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V7
    },
  )
  write_i32(storage, 12, length)
  write_u32(storage, 16, if extensions { 2 } else { 0 })
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
  write_i32(storage, 248, model.sliding_window_tokens())
  if extensions {
    write_u32(
      storage,
      252,
      match kernel.sampling_runtime() {
        HostSamplingRuntime => 0U
        EmbeddedCudaGreedyV1 => 1U
      },
    )
    write_i32(
      storage,
      256,
      if kernel.fused_runtime_source() is Some(_) {
        lengths[5]
      } else {
        0
      },
    )
    write_u32(storage, 260, 0U)
    match kernel.fused_runtime_source() {
      Some(source) =>
        write_bootstrap_source_digest(storage, 264, source.digest_sha256())
      None => ()
    }
  } else {
    write_u32(storage, 252, 0)
    write_u64(storage, 256, 0UL)
  }
  let values = [
    roots.model_root(),
    model.config_locator(),
    model.numeric_weights_locator(),
    roots.kernel_root(),
    kernel.manifest_locator(),
  ]
  let mut cursor = header_bytes
  for value in values {
    cursor = write_bootstrap_source_string(storage, cursor, value)
  }
  match kernel.fused_runtime_source() {
    Some(source) =>
      cursor = write_bootstrap_source_string(storage, cursor, source.locator())
    None => ()
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
    recipe: if extensions {
      DenseMistralBf16PagedAotExtensionsV11
    } else {
      DenseMistralBf16PagedAotV7
    },
    tensor_parallel_policy: None,
    i8_model_source: None,
    i8_execution: None,
    mistral_model_source: Some(model),
    fp8_model_source: None,
    fp8_execution: None,
  }
}

///|
pub fn decode_mistral_bootstrap_source_v7(
  source : FixedArray[Byte],
  length : Int,
) -> EncodedBootstrapSource raise WorkerWireError {
  let extensions = read_u32(source, 4) == MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11
  let header_bytes = if extensions {
    MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V11
  } else {
    MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V7
  }
  if source.length() < header_bytes ||
    length < header_bytes + BOOTSTRAP_SOURCE_DIGEST_BYTES ||
    length > BOOTSTRAP_SOURCE_CAPACITY ||
    read_u32(source, 0) != BOOTSTRAP_SOURCE_MAGIC ||
    (if extensions {
      read_i32(source, 8) != MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V11
    } else {
      read_u32(source, 4) != MISTRAL_BOOTSTRAP_SOURCE_VERSION_V7 ||
      read_i32(source, 8) != MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V7
    }) ||
    read_i32(source, 12) != length ||
    read_u32(source, 16) != (if extensions { 2 } else { 0 }) ||
    (if extensions {
      read_u32(source, 252) > 1U ||
      read_i32(source, 256) < 0 ||
      read_i32(source, 256) > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
      read_u32(source, 260) != 0U ||
      (read_u32(source, 252) == 0U && read_i32(source, 256) == 0)
    } else {
      read_u32(source, 252) != 0 || read_u64(source, 256) != 0UL
    }) {
    raise InvalidBootstrapSource(Reserved)
  }
  let fused = extensions && read_i32(source, 256) > 0
  let value_count = if fused { 6 } else { 5 }
  let lengths = FixedArray::make(value_count, 0)
  let mut payload = 0
  for index in 0..<value_count {
    let item = if index == 5 {
      read_i32(source, 256)
    } else {
      read_i32(source, 20 + index * 4)
    }
    if item <= 0 ||
      item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
      payload > BOOTSTRAP_SOURCE_CAPACITY - item {
      raise InvalidBootstrapSource(Length)
    }
    lengths[index] = item
    payload += item
  }
  if header_bytes + payload + BOOTSTRAP_SOURCE_DIGEST_BYTES != length {
    raise InvalidBootstrapSource(Length)
  }
  let values = FixedArray::make(value_count, "")
  let mut cursor = header_bytes
  for index in 0..<value_count {
    values[index] = read_bootstrap_source_string(source, cursor, lengths[index])
    cursor += lengths[index]
  }
  let roots = WorkerBootstrapRoots::new(
    model_root=values[0],
    kernel_root=values[3],
  )
  let model = WorkerMistralBootstrapModelSource::new(
    config_locator=values[1],
    numeric_weights_locator=values[2],
    config_digest_sha256=bootstrap_source_raw_digest(source, 120).as_hex(),
    numeric_weight_artifact_sha256=bootstrap_source_raw_digest(source, 184).as_hex(),
    weight_manifest_sha256=bootstrap_source_raw_digest(source, 216).as_hex(),
    sliding_window_tokens=read_i32(source, 248),
  )
  let kernel = WorkerBootstrapKernelSource::new(
    manifest_locator=values[4],
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 152).as_hex(),
    fused_runtime=if fused {
      Some(
        WorkerBootstrapFusedRuntimeSource::new(
          locator=values[5],
          digest_sha256=bootstrap_source_raw_digest(source, 264).as_hex(),
        ),
      )
    } else {
      None
    },
    sampling_runtime=if extensions && read_u32(source, 252) == 1U {
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
  let canonical = admit_mistral_bootstrap_source_v7(
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
