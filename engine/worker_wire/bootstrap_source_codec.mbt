///|
const BOOTSTRAP_SOURCE_MAGIC : UInt = 0x5342464c

///|
const BOOTSTRAP_SOURCE_VERSION_V2 : UInt = 2

///|
const BOOTSTRAP_SOURCE_VERSION_V3 : UInt = 3

///|
const BOOTSTRAP_SOURCE_VERSION_V5 : UInt = 5

///|
const BOOTSTRAP_SOURCE_VERSION_V8 : UInt = 8

///|
const BOOTSTRAP_SOURCE_VERSION_V10 : UInt = 10

///|
const BOOTSTRAP_SOURCE_RECIPE_V2 : Int = 2

///|
const BOOTSTRAP_SOURCE_RECIPE_V3 : Int = 3

///|
const BOOTSTRAP_SOURCE_RECIPE_V5 : Int = 5

///|
const BOOTSTRAP_SOURCE_RECIPE_V8 : Int = 8

///|
const BOOTSTRAP_SOURCE_RECIPE_V10 : Int = 10

///|
const BOOTSTRAP_SOURCE_HEADER_BYTES_V2 : Int = 184

///|
const BOOTSTRAP_SOURCE_HEADER_BYTES_V3 : Int = 312

///|
const BOOTSTRAP_SOURCE_HEADER_BYTES_V10 : Int = 224

///|
const BOOTSTRAP_SOURCE_DIGEST_BYTES : Int = 32

///|
/// Shared maximum for legacy, tensor-parallel, and production-I8 sources.
const BOOTSTRAP_SOURCE_CAPACITY : Int = 20744

///|
pub fn bootstrap_source_capacity() -> Int {
  BOOTSTRAP_SOURCE_CAPACITY
}

///|
fn bootstrap_source_header_bytes(kernel : WorkerBootstrapKernelSource) -> Int {
  if kernel.fused_runtime is Some(_) ||
    kernel.sampling_runtime == EmbeddedCudaGreedyV1 {
    BOOTSTRAP_SOURCE_HEADER_BYTES_V10
  } else if kernel.luna_approval is Some(_) {
    BOOTSTRAP_SOURCE_HEADER_BYTES_V3
  } else {
    BOOTSTRAP_SOURCE_HEADER_BYTES_V2
  }
}

///|
fn bootstrap_source_hex_nibble(unit : UInt16) -> Byte raise WorkerWireError {
  if unit >= 48 && unit <= 57 {
    (unit.to_int() - '0'.to_int()).to_byte()
  } else if unit >= 97 && unit <= 102 {
    (unit.to_int() - 'a'.to_int() + 10).to_byte()
  } else {
    raise InvalidBootstrapSource(Digest)
  }
}

///|
fn write_bootstrap_source_digest(
  destination : FixedArray[Byte],
  offset : Int,
  digest : StringView,
) -> Unit raise WorkerWireError {
  if !bootstrap_source_is_sha256(digest) {
    raise InvalidBootstrapSource(Digest)
  }
  for index in 0..<32 {
    destination[offset + index] = (
        bootstrap_source_hex_nibble(digest[index * 2]) << 4
      ) |
      bootstrap_source_hex_nibble(digest[index * 2 + 1])
  }
}

///|
fn bootstrap_source_digest_of(
  bytes : FixedArray[Byte],
  length : Int,
) -> WorkerBootstrapSourceDigest raise WorkerWireError {
  let snapshot = Bytes::makei(length, index => bytes[index])
  WorkerBootstrapSourceDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(snapshot)),
  )
}

///|
fn bootstrap_source_raw_digest(
  bytes : FixedArray[Byte],
  offset : Int,
) -> WorkerBootstrapSourceDigest raise WorkerWireError {
  WorkerBootstrapSourceDigest::from_sha256(
    @crypto.bytes_to_hex_string(
      Bytes::makei(32, index => bytes[offset + index]),
    ),
  )
}

///|
fn write_bootstrap_source_string(
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
fn read_bootstrap_source_string(
  source : FixedArray[Byte],
  cursor : Int,
  byte_count : Int,
) -> String raise WorkerWireError {
  @utf8.decode(Bytes::makei(byte_count, index => source[cursor + index])) catch {
    _ => raise InvalidBootstrapSource(Utf8)
  }
}

///|
fn bootstrap_source_lengths(
  roots : WorkerBootstrapRoots,
  model : WorkerBootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
) -> Array[Int] {
  let lengths = [
    @utf8.encode(roots.model_root).length(),
    @utf8.encode(model.config_locator).length(),
    @utf8.encode(model.weights_locator).length(),
    @utf8.encode(roots.kernel_root).length(),
    @utf8.encode(kernel.manifest_locator).length(),
  ]
  match kernel.fused_runtime {
    Some(source) => lengths.push(@utf8.encode(source.locator).length())
    None => ()
  }
  lengths
}

///|
/// Owns a canonical, bounded source snapshot from five focused immutable
/// records. No filesystem read or device operation occurs at this boundary.
pub fn admit_bootstrap_source(
  roots : WorkerBootstrapRoots,
  model : WorkerBootstrapModelSource,
  kernel : WorkerBootstrapKernelSource,
  execution : WorkerBootstrapExecution,
  ceilings : WorkerBootstrapCeilings,
) -> EncodedBootstrapSource raise WorkerWireError {
  let lengths = bootstrap_source_lengths(roots, model, kernel)
  if kernel.luna_approval is Some(_) &&
    (
      kernel.fused_runtime is Some(_) ||
      kernel.sampling_runtime == EmbeddedCudaGreedyV1
    ) {
    raise InvalidBootstrapSource(Recipe)
  }
  let header_bytes = bootstrap_source_header_bytes(kernel)
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
  let recipe = if kernel.fused_runtime is Some(_) ||
    kernel.sampling_runtime == EmbeddedCudaGreedyV1 {
    DenseLlamaBf16PagedAotExtensionsV10
  } else if kernel.luna_approval is Some(_) {
    DenseLlamaPagedAotV3
  } else {
    DenseLlamaPagedAotV2
  }
  write_u32(storage, 0, BOOTSTRAP_SOURCE_MAGIC)
  write_u32(
    storage,
    4,
    if recipe == DenseLlamaBf16PagedAotExtensionsV10 {
      BOOTSTRAP_SOURCE_VERSION_V10
    } else if recipe == DenseLlamaPagedAotV3 {
      BOOTSTRAP_SOURCE_VERSION_V3
    } else {
      BOOTSTRAP_SOURCE_VERSION_V2
    },
  )
  write_i32(
    storage,
    8,
    if recipe == DenseLlamaBf16PagedAotExtensionsV10 {
      BOOTSTRAP_SOURCE_RECIPE_V10
    } else if recipe == DenseLlamaPagedAotV3 {
      BOOTSTRAP_SOURCE_RECIPE_V3
    } else {
      BOOTSTRAP_SOURCE_RECIPE_V2
    },
  )
  write_i32(storage, 12, length)
  write_u32(
    storage,
    16,
    if recipe == DenseLlamaPagedAotV3 {
      1
    } else if recipe == DenseLlamaBf16PagedAotExtensionsV10 {
      2
    } else {
      0
    },
  )
  for index in 0..<5 {
    write_i32(storage, 20 + index * 4, lengths[index])
  }
  if recipe == DenseLlamaBf16PagedAotExtensionsV10 {
    write_i32(
      storage,
      216,
      if kernel.fused_runtime is Some(_) {
        lengths[5]
      } else {
        0
      },
    )
    write_u32(
      storage,
      220,
      match kernel.sampling_runtime {
        HostSamplingRuntime => 0U
        EmbeddedCudaGreedyV1 => 1U
      },
    )
  }
  write_i32(storage, 40, execution.compute_major)
  write_i32(storage, 44, execution.compute_minor)
  let flags : UInt = (if execution.supports_bf16 { 1 } else { 0 }) |
    (if execution.supports_cublas_lt { 2 } else { 0 })
  write_u32(storage, 48, flags)
  write_i32(storage, 52, execution.tokens_per_page)
  write_i32(storage, 56, execution.total_page_count)
  write_i32(storage, 60, ceilings.max_model_config_bytes)
  write_u64(storage, 64, ceilings.max_weight_file_bytes.reinterpret_as_uint64())
  write_u64(
    storage,
    72,
    ceilings.max_weight_arena_bytes.reinterpret_as_uint64(),
  )
  write_u64(
    storage,
    80,
    ceilings.max_activation_arena_bytes.reinterpret_as_uint64(),
  )
  write_u64(storage, 88, ceilings.max_kv_arena_bytes.reinterpret_as_uint64())
  write_u64(
    storage,
    96,
    ceilings.max_execution_manifest_bytes.reinterpret_as_uint64(),
  )
  write_u64(storage, 104, ceilings.max_module_bytes.reinterpret_as_uint64())
  write_u64(
    storage,
    112,
    ceilings.max_total_module_bytes.reinterpret_as_uint64(),
  )
  write_bootstrap_source_digest(storage, 120, model.config_digest)
  write_bootstrap_source_digest(storage, 152, kernel.manifest_digest)
  match kernel.luna_approval {
    Some(approval) => {
      write_bootstrap_source_digest(storage, 184, approval.manifest_digest)
      write_bootstrap_source_digest(storage, 216, approval.key_identity)
      write_bootstrap_source_digest(storage, 248, approval.signature_digest)
      write_bootstrap_source_digest(storage, 280, approval.approved_source)
    }
    None => ()
  }
  match kernel.fused_runtime {
    Some(source) => write_bootstrap_source_digest(storage, 184, source.digest)
    None => ()
  }
  let mut cursor = header_bytes
  cursor = write_bootstrap_source_string(storage, cursor, roots.model_root)
  cursor = write_bootstrap_source_string(storage, cursor, model.config_locator)
  cursor = write_bootstrap_source_string(storage, cursor, model.weights_locator)
  cursor = write_bootstrap_source_string(storage, cursor, roots.kernel_root)
  cursor = write_bootstrap_source_string(
    storage,
    cursor,
    kernel.manifest_locator,
  )
  match kernel.fused_runtime {
    Some(source) =>
      cursor = write_bootstrap_source_string(storage, cursor, source.locator)
    None => ()
  }
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
    recipe,
    tensor_parallel_policy: None,
    i8_model_source: None,
    i8_execution: None,
    mistral_model_source: None,
    fp8_model_source: None,
    fp8_execution: None,
  }
}

///|
pub fn EncodedBootstrapSource::copy_to(
  self : EncodedBootstrapSource,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise WorkerWireError {
  if destination_offset < 0 ||
    destination_offset > destination.length() - self.length {
    raise DestinationTooSmall
  }
  for index in 0..<self.length {
    destination[destination_offset + index] = self.storage[index]
  }
}
