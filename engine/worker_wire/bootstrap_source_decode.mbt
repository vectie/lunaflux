///|
fn validate_bootstrap_source_header(
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise WorkerWireError {
  if source.length() < 20 {
    raise InvalidBootstrapSource(Length)
  }
  let version = read_u32(source, 4)
  let recipe = read_i32(source, 8)
  if version != BOOTSTRAP_SOURCE_VERSION_V2 &&
    version != BOOTSTRAP_SOURCE_VERSION_V3 &&
    version != BOOTSTRAP_SOURCE_VERSION_V5 &&
    version != I8_BOOTSTRAP_SOURCE_VERSION_V6 &&
    version != MISTRAL_BOOTSTRAP_SOURCE_VERSION_V7 &&
    version != BOOTSTRAP_SOURCE_VERSION_V8 &&
    version != FP8_BOOTSTRAP_SOURCE_VERSION_V9 &&
    version != BOOTSTRAP_SOURCE_VERSION_V10 &&
    version != MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11 &&
    version != QWEN3_BOOTSTRAP_SOURCE_VERSION_V12 {
    raise InvalidBootstrapSource(Version)
  }
  let header_bytes = if version == BOOTSTRAP_SOURCE_VERSION_V2 {
    if recipe != BOOTSTRAP_SOURCE_RECIPE_V2 {
      raise InvalidBootstrapSource(Recipe)
    }
    BOOTSTRAP_SOURCE_HEADER_BYTES_V2
  } else if version == BOOTSTRAP_SOURCE_VERSION_V3 {
    if recipe != BOOTSTRAP_SOURCE_RECIPE_V3 {
      raise InvalidBootstrapSource(Recipe)
    }
    BOOTSTRAP_SOURCE_HEADER_BYTES_V3
  } else if version == BOOTSTRAP_SOURCE_VERSION_V5 {
    if recipe != BOOTSTRAP_SOURCE_RECIPE_V5 || source.length() < 208 {
      raise InvalidBootstrapSource(Recipe)
    }
    let rank_count = read_i32(source, 184)
    if rank_count < 2 ||
      rank_count > 16 ||
      read_u32(source, 196) != 0U ||
      read_u32(source, 204) != 0U {
      raise InvalidBootstrapSource(Reserved)
    }
    208 + rank_count * 48
  } else if version == BOOTSTRAP_SOURCE_VERSION_V8 {
    if recipe != BOOTSTRAP_SOURCE_RECIPE_V8 || source.length() < 336 {
      raise InvalidBootstrapSource(Recipe)
    }
    let rank_count = read_i32(source, 184)
    if rank_count < 2 ||
      rank_count > 16 ||
      read_u32(source, 196) != 0U ||
      read_u32(source, 204) != 0U {
      raise InvalidBootstrapSource(Reserved)
    }
    336 + rank_count * 48
  } else if version == I8_BOOTSTRAP_SOURCE_VERSION_V6 {
    if version != I8_BOOTSTRAP_SOURCE_VERSION_V6 ||
      recipe != I8_BOOTSTRAP_SOURCE_RECIPE_V6 {
      raise InvalidBootstrapSource(Recipe)
    }
    I8_BOOTSTRAP_SOURCE_HEADER_BYTES_V6
  } else if version == FP8_BOOTSTRAP_SOURCE_VERSION_V9 {
    if recipe != FP8_BOOTSTRAP_SOURCE_RECIPE_V9 {
      raise InvalidBootstrapSource(Recipe)
    }
    FP8_BOOTSTRAP_SOURCE_HEADER_BYTES_V9
  } else if version == BOOTSTRAP_SOURCE_VERSION_V10 {
    let fused_length = read_i32(source, 216)
    let sampling = read_u32(source, 220)
    if recipe != BOOTSTRAP_SOURCE_RECIPE_V10 ||
      source.length() < BOOTSTRAP_SOURCE_HEADER_BYTES_V10 ||
      fused_length < 0 ||
      fused_length > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
      sampling > 1U ||
      (fused_length == 0 && sampling == 0U) {
      raise InvalidBootstrapSource(Recipe)
    }
    BOOTSTRAP_SOURCE_HEADER_BYTES_V10
  } else if version == MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11 {
    if recipe != MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V11 {
      raise InvalidBootstrapSource(Recipe)
    }
    MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V11
  } else if version == QWEN3_BOOTSTRAP_SOURCE_VERSION_V12 {
    if recipe != QWEN3_BOOTSTRAP_SOURCE_RECIPE_V12 {
      raise InvalidBootstrapSource(Recipe)
    }
    QWEN3_BOOTSTRAP_SOURCE_HEADER_BYTES_V12
  } else {
    if recipe != MISTRAL_BOOTSTRAP_SOURCE_RECIPE_V7 {
      raise InvalidBootstrapSource(Recipe)
    }
    MISTRAL_BOOTSTRAP_SOURCE_HEADER_BYTES_V7
  }
  if length < header_bytes + BOOTSTRAP_SOURCE_DIGEST_BYTES ||
    length > BOOTSTRAP_SOURCE_CAPACITY ||
    source.length() < length ||
    read_i32(source, 12) != length {
    raise InvalidBootstrapSource(Length)
  }
  if read_u32(source, 0) != BOOTSTRAP_SOURCE_MAGIC {
    raise InvalidBootstrapSource(Magic)
  }
  if read_u32(source, 16) !=
    (if version == BOOTSTRAP_SOURCE_VERSION_V3 ||
      version == BOOTSTRAP_SOURCE_VERSION_V8 ||
      version == FP8_BOOTSTRAP_SOURCE_VERSION_V9 {
      1
    } else if version == BOOTSTRAP_SOURCE_VERSION_V10 {
      2
    } else if version == MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11 {
      2
    } else {
      0
    }) ||
    (read_u32(source, 48) & 0xfffffffc) != 0 {
    raise InvalidBootstrapSource(Reserved)
  }
  let expected = bootstrap_source_digest_of(
    source,
    length - BOOTSTRAP_SOURCE_DIGEST_BYTES,
  )
  if bootstrap_source_raw_digest(source, length - BOOTSTRAP_SOURCE_DIGEST_BYTES) !=
    expected {
    raise InvalidBootstrapSource(Integrity)
  }
}

///|
fn bootstrap_source_read_i64(source : FixedArray[Byte], offset : Int) -> Int64 {
  read_u64(source, offset).reinterpret_as_int64()
}

///|
/// Validates and owns one untrusted canonical bootstrap-source frame. This
/// proves internal integrity and lexical path syntax only; it does not grant
/// filesystem authority or establish that a root is stable or read-only.
/// Rejected inputs publish no partially decoded value and retain no alias to
/// `source`.
pub fn decode_bootstrap_source(
  source : FixedArray[Byte],
  length : Int,
) -> EncodedBootstrapSource raise WorkerWireError {
  validate_bootstrap_source_header(source, length)
  if read_u32(source, 4) == BOOTSTRAP_SOURCE_VERSION_V5 ||
    read_u32(source, 4) == BOOTSTRAP_SOURCE_VERSION_V8 {
    return decode_tensor_parallel_bootstrap_source(source, length)
  }
  if read_u32(source, 4) == I8_BOOTSTRAP_SOURCE_VERSION_V6 {
    return decode_i8_bootstrap_source_v6(source, length)
  }
  if read_u32(source, 4) == MISTRAL_BOOTSTRAP_SOURCE_VERSION_V7 ||
    read_u32(source, 4) == MISTRAL_BOOTSTRAP_SOURCE_VERSION_V11 {
    return decode_mistral_bootstrap_source_v7(source, length)
  }
  if read_u32(source, 4) == QWEN3_BOOTSTRAP_SOURCE_VERSION_V12 {
    return decode_qwen3_bf16_paged_bootstrap_source_v12(source, length)
  }
  if read_u32(source, 4) == FP8_BOOTSTRAP_SOURCE_VERSION_V9 {
    return decode_fp8_reusable_bootstrap_source_v9(source, length)
  }
  let version = read_u32(source, 4)
  let specialized = version == BOOTSTRAP_SOURCE_VERSION_V3
  let extensions = version == BOOTSTRAP_SOURCE_VERSION_V10
  let fused = extensions && read_i32(source, 216) > 0
  let header_bytes = if extensions {
    BOOTSTRAP_SOURCE_HEADER_BYTES_V10
  } else if specialized {
    BOOTSTRAP_SOURCE_HEADER_BYTES_V3
  } else {
    BOOTSTRAP_SOURCE_HEADER_BYTES_V2
  }
  let value_count = if fused { 6 } else { 5 }
  let lengths = FixedArray::make(value_count, 0)
  let mut payload_bytes = 0
  for index in 0..<value_count {
    let item = if index == 5 {
      read_i32(source, 216)
    } else {
      read_i32(source, 20 + index * 4)
    }
    if item <= 0 || item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES {
      raise InvalidBootstrapSource(Length)
    }
    if payload_bytes > BOOTSTRAP_SOURCE_CAPACITY - item {
      raise InvalidBootstrapSource(Length)
    }
    lengths[index] = item
    payload_bytes += item
  }
  if header_bytes + payload_bytes + BOOTSTRAP_SOURCE_DIGEST_BYTES != length {
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
  let model = WorkerBootstrapModelSource::new(
    config_locator=values[1],
    weights_locator=values[2],
    config_digest_sha256=bootstrap_source_raw_digest(source, 120).as_hex(),
  )
  let kernel = WorkerBootstrapKernelSource::new(
    manifest_locator=values[4],
    manifest_digest_sha256=bootstrap_source_raw_digest(source, 152).as_hex(),
    luna_approval=if specialized {
      Some(
        WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
          manifest_digest_sha256=bootstrap_source_raw_digest(source, 184).as_hex(),
          key_identity_sha256=bootstrap_source_raw_digest(source, 216).as_hex(),
          signature_digest_sha256=bootstrap_source_raw_digest(source, 248).as_hex(),
          approved_source_sha256=bootstrap_source_raw_digest(source, 280).as_hex(),
        ),
      )
    } else {
      None
    },
    fused_runtime=if fused {
      Some(
        WorkerBootstrapFusedRuntimeSource::new(
          locator=values[5],
          digest_sha256=bootstrap_source_raw_digest(source, 184).as_hex(),
        ),
      )
    } else {
      None
    },
    sampling_runtime=if extensions && read_u32(source, 220) == 1U {
      EmbeddedCudaGreedyV1
    } else {
      HostSamplingRuntime
    },
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
  let canonical = admit_bootstrap_source(
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
