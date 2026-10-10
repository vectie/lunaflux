///|
fn map_file_error(
  shard : Int,
  error : @approved_fs.ApprovedFsError,
) -> StreamingSafetensorsError {
  match error {
    UnsupportedType(_) => UnsupportedFileType(shard)
    Truncated => FileTruncated(shard)
    FileTooLarge => FileTooLarge(shard)
    ChangedDuringRead => FileChanged(shard)
    _ => FileUnavailable(shard)
  }
}

///|
fn close_file(
  shard : Int,
  file : @approved_fs.ApprovedFile,
) -> Unit raise StreamingSafetensorsError {
  file.close() catch {
    _ => raise FileCloseFailed(shard)
  }
}

///|
fn[T] with_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  shard : Int,
  body : (@approved_fs.ApprovedFile) -> T raise StreamingSafetensorsError,
) -> T raise StreamingSafetensorsError {
  let file = root.open_file(locator) catch {
    error => raise map_file_error(shard, error)
  }
  let outcome : Result[T, StreamingSafetensorsError] = Ok(body(file)) catch {
    error => Err(error)
  }
  match outcome {
    Err(primary) => {
      close_file(shard, file) catch {
        _ => ()
      }
      raise primary
    }
    Ok(value) => {
      close_file(shard, file)
      value
    }
  }
}

///|
fn read_exactly(
  shard : Int,
  file : @approved_fs.ApprovedFile,
  destination : FixedArray[Byte],
  destination_offset : Int,
  byte_count : Int,
  position : Int64,
) -> Unit raise StreamingSafetensorsError {
  file.read_exactly_at(destination, destination_offset~, byte_count~, position~) catch {
    error => raise map_file_error(shard, error)
  }
}

///|
fn stamp_file(
  shard : Int,
  file : @approved_fs.ApprovedFile,
) -> @approved_fs.ApprovedFileStamp raise StreamingSafetensorsError {
  file.stamp() catch {
    error => raise map_file_error(shard, error)
  }
}

///|
fn validate_dtype_specs(
  specs : ArrayView[StreamingSafetensorsDTypeSpec],
  limits : StreamingSafetensorsLimits,
) -> Unit raise StreamingSafetensorsError {
  if specs.is_empty() || specs.length() > limits.maximum_dtype_specs {
    raise InvalidDTypeSpec
  }
  for index, spec in specs {
    if !valid_ascii_token(spec.tag, 64) || spec.bytes_per_element <= 0L {
      raise InvalidDTypeSpec
    }
    for earlier in 0..<index {
      if specs[earlier].tag == spec.tag {
        raise DuplicateDTypeTag(earlier, index)
      }
    }
  }
}

///|
fn inspect_open_file(
  shard : Int,
  file : @approved_fs.ApprovedFile,
  input : StreamingSafetensorsShardInput,
  specs : ArrayView[StreamingSafetensorsDTypeSpec],
  limits : StreamingSafetensorsLimits,
) -> InspectedShard raise StreamingSafetensorsError {
  let initial_stamp = stamp_file(shard, file)
  let file_bytes = initial_stamp.byte_count()
  if file_bytes <= STREAMING_SAFETENSORS_PREFIX_BYTES.to_int64() {
    raise FileTruncated(shard)
  }
  if file_bytes > limits.maximum_file_bytes {
    raise FileTooLarge(shard)
  }
  let prefix = FixedArray::make(STREAMING_SAFETENSORS_PREFIX_BYTES, b'\x00')
  read_exactly(shard, file, prefix, 0, prefix.length(), 0L)
  let declared = decode_header_length(prefix)
  if declared > 2147483647UL {
    raise HeaderLengthOverflow(shard)
  }
  let header_bytes = declared.to_int()
  if header_bytes <= 0 || header_bytes > limits.maximum_header_bytes {
    raise HeaderTooLarge(shard)
  }
  let payload_offset = STREAMING_SAFETENSORS_PREFIX_BYTES.to_int64() +
    header_bytes.to_int64()
  if payload_offset > file_bytes {
    raise TruncatedHeader(shard)
  }
  let payload_bytes = file_bytes - payload_offset
  let header = FixedArray::make(header_bytes, b'\x00')
  read_exactly(
    shard,
    file,
    header,
    0,
    header_bytes,
    STREAMING_SAFETENSORS_PREFIX_BYTES.to_int64(),
  )
  let tensors = parse_header(header, shard, payload_bytes, specs, limits)
  let final_stamp = stamp_file(shard, file)
  if final_stamp != initial_stamp {
    raise FileChanged(shard)
  }
  {
    locator: input.locator,
    stamp: initial_stamp,
    metadata: {
      file_byte_count: file_bytes,
      payload_offset,
      payload_byte_count: payload_bytes,
      file_digest: input.expected_digest,
      tensors: ReadOnlyArray::from_array(tensors),
    },
  }
}

///|
/// Read bounded headers and close each shard before publishing metadata.
/// Supplied digests are inventory labels, not verified payload identities.
pub fn inspect_shards(
  root : @approved_fs.ApprovedRoot,
  inputs : ArrayView[StreamingSafetensorsShardInput],
  dtype_specs~ : ArrayView[StreamingSafetensorsDTypeSpec],
  limits~ : StreamingSafetensorsLimits,
) -> StreamingSafetensorsInspection raise StreamingSafetensorsError {
  if inputs.is_empty() || inputs.length() > limits.maximum_shards {
    raise InvalidLimit
  }
  validate_dtype_specs(dtype_specs, limits)
  let shards : Array[InspectedShard] = []
  let names : Map[String, (Int, Int)] = Map([])
  let mut total = 0
  for shard_ordinal, input in inputs {
    let shard = with_file(root, input.locator, shard_ordinal, file => {
      inspect_open_file(shard_ordinal, file, input, dtype_specs, limits)
    })
    for tensor_ordinal, tensor in shard.metadata.tensors {
      if names.get(tensor.name) is Some((first_shard, first_tensor)) {
        raise DuplicateTensorName(
          first_shard~,
          first_tensor~,
          second_shard=shard_ordinal,
          second_tensor=tensor_ordinal,
        )
      }
      names[tensor.name] = (shard_ordinal, tensor_ordinal)
    }
    if total > limits.maximum_total_tensors - shard.metadata.tensors.length() {
      raise TooManyTensors(shard_ordinal)
    }
    total += shard.metadata.tensors.length()
    shards.push(shard)
  }
  { shards: ReadOnlyArray::from_array(shards), total_tensor_count: total, }
}
