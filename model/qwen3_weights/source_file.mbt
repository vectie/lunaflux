///|
const QWEN3_SAFETENSORS_PREFIX_BYTES : Int = 8

///|
fn qwen3_open_source_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
) -> @approved_fs.ApprovedFile raise Qwen3WeightsError {
  root.open_file(locator) catch {
    _ => raise SourceFileUnavailable
  }
}

///|
fn qwen3_read_source_range(
  file : @approved_fs.ApprovedFile,
  destination : FixedArray[Byte],
  destination_offset : Int,
  position : Int64,
  byte_count : Int,
) -> Unit raise Qwen3WeightsError {
  file.read_exactly_at(destination, destination_offset~, position~, byte_count~) catch {
    _ => raise SourceFileUnavailable
  }
}

///|
fn qwen3_source_stamp(
  file : @approved_fs.ApprovedFile,
) -> @approved_fs.ApprovedFileStamp raise Qwen3WeightsError {
  file.stamp() catch {
    _ => raise SourceFileUnavailable
  }
}

///|
fn qwen3_close_source_file(
  file : @approved_fs.ApprovedFile,
) -> Unit raise Qwen3WeightsError {
  file.close() catch {
    _ => raise SourceFileCloseFailed
  }
}

///|
fn qwen3_hash_source_file(
  file : @approved_fs.ApprovedFile,
  file_bytes : Int64,
  scratch : FixedArray[Byte],
) -> String raise Qwen3WeightsError {
  let digest = @crypto.SHA256::new()
  let mut position = 0L
  while position < file_bytes {
    let remaining = file_bytes - position
    let count = if remaining < scratch.length().to_int64() {
      remaining.to_int()
    } else {
      scratch.length()
    }
    qwen3_read_source_range(file, scratch, 0, position, count)
    digest.update_from_iter(scratch[:count].iter())
    position += count.to_int64()
  }
  @crypto.bytes_to_hex_string(digest.finalize())
}

///|
fn qwen3_ranges_equal(
  file : @approved_fs.ApprovedFile,
  left_start : Int64,
  right_start : Int64,
  byte_length : Int64,
  left : FixedArray[Byte],
  right : FixedArray[Byte],
) -> Bool raise Qwen3WeightsError {
  let mut position = 0L
  while position < byte_length {
    let remaining = byte_length - position
    let count = if remaining < left.length().to_int64() {
      remaining.to_int()
    } else {
      left.length()
    }
    qwen3_read_source_range(file, left, 0, left_start + position, count)
    qwen3_read_source_range(file, right, 0, right_start + position, count)
    for index in 0..<count {
      if left[index] != right[index] {
        return false
      }
    }
    position += count.to_int64()
  }
  true
}

///|
fn qwen3_source_manifest_sha256(
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  bindings : Qwen3WeightBindings,
  artifact_sha256 : StringView,
) -> String {
  let encoded = StringBuilder()
  encoded.write_string(
    "schema=lunaflux.qwen3-source-manifest.v1\ncontent=" +
    metadata.content_digest().as_hex() +
    "\nartifact=" +
    artifact_sha256.to_owned() +
    "\ntensor_count=" +
    bindings.source_tensor_count().to_string() +
    "\nmaterialized_bytes=" +
    bindings.materialized_bytes().to_string() +
    "\ntied_lm_head_tensor_ref=0\nredundant_lm_head_byte_equal=" +
    (if bindings.redundant_lm_head_tensor() is Some(_) {
      "true"
    } else {
      "absent"
    }) +
    "\n",
  )
  for reference in 0..<bindings.length() {
    guard bindings.binding(try! @plan.TensorRef::new(reference))
      is Some(binding) else {
      abort("validated Qwen3 source binding invariant")
    }
    let tensor = binding.tensor()
    let dimensions : Array[String] = []
    for dimension in tensor.shape() {
      dimensions.push(dimension.to_string())
    }
    encoded.write_string(
      "tensor=" +
      reference.to_string() +
      "|" +
      tensor.name() +
      "|BF16|" +
      dimensions.join(",") +
      "|" +
      tensor.data_start().to_string() +
      "|" +
      tensor.data_end().to_string() +
      "\n",
    )
  }
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(encoded.to_string())))
}

///|
fn qwen3_inspect_open_source(
  file : @approved_fs.ApprovedFile,
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  expected_artifact_sha256 : StringView,
  limits : Qwen3SourceFileLimits,
) -> Qwen3SourceFileInspection raise Qwen3WeightsError {
  if expected_artifact_sha256.length() != 64 ||
    !expected_artifact_sha256.all(ch => {
      ch.is_ascii_digit() || (ch >= 'a' && ch <= 'f')
    }) {
    raise SourceArtifactDigestMismatch
  }
  let initial_stamp = qwen3_source_stamp(file)
  let file_bytes = initial_stamp.byte_count()
  if file_bytes <= QWEN3_SAFETENSORS_PREFIX_BYTES.to_int64() {
    raise SourceFileUnavailable
  }
  if file_bytes > limits.max_file_bytes {
    raise SourceFileTooLarge
  }
  let prefix = FixedArray::make(QWEN3_SAFETENSORS_PREFIX_BYTES, b'\x00')
  qwen3_read_source_range(file, prefix, 0, 0L, prefix.length())
  let header_bytes = @safetensors_reader.declared_header_bytes(
    Bytes::from_array(prefix[:]),
    limits=limits.safetensors,
  ) catch {
    error => raise InvalidSourceSafetensors(error)
  }
  if header_bytes > 2147483639 {
    raise InvalidSourceSafetensors(HeaderLengthOverflow)
  }
  let payload_offset_bytes = QWEN3_SAFETENSORS_PREFIX_BYTES.to_int64() +
    header_bytes.to_int64()
  if payload_offset_bytes > file_bytes {
    raise InvalidSourceSafetensors(TruncatedHeader)
  }
  let header_scratch = FixedArray::make(payload_offset_bytes.to_int(), b'\x00')
  qwen3_read_source_range(file, header_scratch, 0, 0L, header_scratch.length())
  let tensors = @safetensors_reader.read_header(
    Bytes::from_array(header_scratch[:]),
    payload_bytes=file_bytes - payload_offset_bytes,
    limits=limits.safetensors,
  ) catch {
    error => raise InvalidSourceSafetensors(error)
  }
  let bindings = bind_qwen3_metadata(metadata, tensors)
  let left = FixedArray::make(limits.compare_chunk_bytes, b'\x00')
  let right = FixedArray::make(limits.compare_chunk_bytes, b'\x00')
  match bindings.redundant_lm_head_tensor() {
    None => ()
    Some(head) => {
      let embedding = bindings.embedding_binding().tensor()
      if !qwen3_ranges_equal(
          file,
          payload_offset_bytes + embedding.data_start(),
          payload_offset_bytes + head.data_start(),
          embedding.byte_length(),
          left,
          right,
        ) {
        raise InvalidTensor(reference=0, TiedOutputByteEquality)
      }
    }
  }
  let artifact_sha256 = qwen3_hash_source_file(file, file_bytes, left)
  if artifact_sha256 != expected_artifact_sha256.to_owned() {
    raise SourceArtifactDigestMismatch
  }
  if qwen3_source_stamp(file) != initial_stamp {
    raise SourceFileChanged
  }
  {
    bindings,
    artifact_sha256,
    source_manifest_sha256: qwen3_source_manifest_sha256(
      metadata, bindings, artifact_sha256,
    ),
  }
}

///|
/// Authenticates one original dense-Qwen3 BF16 safetensors artifact offline.
/// A redundant physical LM head is accepted only after comparing every source
/// byte with the embedding range on the same pinned file handle.
pub fn inspect_source_safetensors_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_sha256~ : StringView,
  limits~ : Qwen3SourceFileLimits,
) -> Qwen3SourceFileInspection raise Qwen3WeightsError {
  let file = qwen3_open_source_file(root, locator)
  let outcome : Result[Qwen3SourceFileInspection, Qwen3WeightsError] = Ok(
    qwen3_inspect_open_source(file, metadata, expected_artifact_sha256, limits),
  ) catch {
    error => Err(error)
  }
  match outcome {
    Err(primary) => {
      qwen3_close_source_file(file) catch {
        _ => ()
      }
      raise primary
    }
    Ok(inspection) => {
      qwen3_close_source_file(file)
      inspection
    }
  }
}
