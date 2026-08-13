///|
const BOOTSTRAP_SOURCE_MAGIC : UInt = 0x5342464c

///|
const BOOTSTRAP_SOURCE_VERSION : UInt = 1

///|
const BOOTSTRAP_SOURCE_RECIPE : Int = 1

///|
const BOOTSTRAP_SOURCE_HEADER_BYTES : Int = 184

///|
const BOOTSTRAP_SOURCE_DIGEST_BYTES : Int = 32

///|
const BOOTSTRAP_SOURCE_CAPACITY : Int = 20696

///|
pub fn bootstrap_source_capacity() -> Int {
  BOOTSTRAP_SOURCE_CAPACITY
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
) -> FixedArray[Int] {
  FixedArray::from_array([
    @utf8.encode(roots.model_root).length(),
    @utf8.encode(model.config_locator).length(),
    @utf8.encode(model.weights_locator).length(),
    @utf8.encode(roots.kernel_root).length(),
    @utf8.encode(kernel.manifest_locator).length(),
  ])
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
  let mut length = BOOTSTRAP_SOURCE_HEADER_BYTES + BOOTSTRAP_SOURCE_DIGEST_BYTES
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
  write_u32(storage, 4, BOOTSTRAP_SOURCE_VERSION)
  write_i32(storage, 8, BOOTSTRAP_SOURCE_RECIPE)
  write_i32(storage, 12, length)
  write_u32(storage, 16, 0)
  for index, item in lengths {
    write_i32(storage, 20 + index * 4, item)
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
  let mut cursor = BOOTSTRAP_SOURCE_HEADER_BYTES
  cursor = write_bootstrap_source_string(storage, cursor, roots.model_root)
  cursor = write_bootstrap_source_string(storage, cursor, model.config_locator)
  cursor = write_bootstrap_source_string(storage, cursor, model.weights_locator)
  cursor = write_bootstrap_source_string(storage, cursor, roots.kernel_root)
  cursor = write_bootstrap_source_string(
    storage,
    cursor,
    kernel.manifest_locator,
  )
  let digest = bootstrap_source_digest_of(storage, cursor)
  write_bootstrap_source_digest(storage, cursor, digest.as_hex())
  { storage, length, digest, roots, model, kernel, execution, ceilings }
}

///|
fn validate_bootstrap_source_header(
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise WorkerWireError {
  if length < BOOTSTRAP_SOURCE_HEADER_BYTES + BOOTSTRAP_SOURCE_DIGEST_BYTES ||
    length > BOOTSTRAP_SOURCE_CAPACITY ||
    source.length() < length ||
    read_i32(source, 12) != length {
    raise InvalidBootstrapSource(Length)
  }
  if read_u32(source, 0) != BOOTSTRAP_SOURCE_MAGIC {
    raise InvalidBootstrapSource(Magic)
  }
  if read_u32(source, 4) != BOOTSTRAP_SOURCE_VERSION {
    raise InvalidBootstrapSource(Version)
  }
  if read_i32(source, 8) != BOOTSTRAP_SOURCE_RECIPE {
    raise InvalidBootstrapSource(Recipe)
  }
  if read_u32(source, 16) != 0 || (read_u32(source, 48) & 0xfffffffc) != 0 {
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
  let lengths = FixedArray::make(5, 0)
  let mut payload_bytes = 0
  for index in 0..<5 {
    let item = read_i32(source, 20 + index * 4)
    if item <= 0 || item > BOOTSTRAP_SOURCE_MAX_PATH_BYTES {
      raise InvalidBootstrapSource(Length)
    }
    if payload_bytes > BOOTSTRAP_SOURCE_CAPACITY - item {
      raise InvalidBootstrapSource(Length)
    }
    lengths[index] = item
    payload_bytes += item
  }
  if BOOTSTRAP_SOURCE_HEADER_BYTES +
    payload_bytes +
    BOOTSTRAP_SOURCE_DIGEST_BYTES !=
    length {
    raise InvalidBootstrapSource(Length)
  }
  let values = FixedArray::make(5, "")
  let mut cursor = BOOTSTRAP_SOURCE_HEADER_BYTES
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
