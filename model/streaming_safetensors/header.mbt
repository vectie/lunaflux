///|
const STREAMING_SAFETENSORS_PREFIX_BYTES : Int = 8

///|
fn decode_header_length(prefix : FixedArray[Byte]) -> UInt64 {
  let mut value = 0UL
  for index in 0..<STREAMING_SAFETENSORS_PREFIX_BYTES {
    value = value | (prefix[index].to_uint64() << (index * 8))
  }
  value
}

///|
fn header_has_valid_envelope(header : FixedArray[Byte]) -> Bool {
  if header.is_empty() || header[0] != b'{' {
    return false
  }
  for index = header.length() - 1; index >= 0; index = index - 1 {
    match header[index] {
      b' ' => continue
      b'}' => return true
      _ => return false
    }
  }
  false
}

///|
fn decode_exact_int64(
  value : Json,
  shard : Int,
  tensor : Int,
  field : StreamingSafetensorsJsonField,
) -> Int64 raise StreamingSafetensorsError {
  guard value is Number(number, repr~) else {
    raise InvalidInteger(shard~, tensor~, field)
  }
  match repr {
    Some(encoded) => {
      let integer : Int64 = @string.parse_int64(encoded) catch {
        _ => raise InvalidInteger(shard~, tensor~, field)
      }
      integer
    }
    None => {
      if number.is_nan() {
        raise InvalidInteger(shard~, tensor~, field)
      }
      let integer = number.to_int64()
      if integer.to_double() != number {
        raise InvalidInteger(shard~, tensor~, field)
      }
      integer
    }
  }
}

///|
fn require_tensor_field(
  fields : Map[String, Json],
  name : String,
  shard : Int,
  tensor : Int,
  field : StreamingSafetensorsJsonField,
) -> Json raise StreamingSafetensorsError {
  match fields.get(name) {
    Some(value) => value
    None => raise MissingTensorField(shard~, tensor~, field)
  }
}

///|
fn dtype_for_tag(
  specs : ArrayView[StreamingSafetensorsDTypeSpec],
  tag : StringView,
) -> StreamingSafetensorsDTypeSpec? {
  for spec in specs {
    if spec.tag[:] == tag {
      return Some(spec)
    }
  }
  None
}

///|
fn decode_shape(
  value : Json,
  shard : Int,
  tensor : Int,
  limits : StreamingSafetensorsLimits,
) -> Array[Int64] raise StreamingSafetensorsError {
  guard value is Array(dimensions) else {
    raise InvalidTensorField(shard~, tensor~, Shape)
  }
  // A rank-zero shape denotes one scalar element (for example ModelOpt's
  // NVFP4 global weight scale). It is not a zero-element tensor.
  if dimensions.length() > limits.maximum_rank {
    raise InvalidRank(shard~, tensor~)
  }
  let shape : Array[Int64] = []
  for dimension in dimensions {
    let decoded = decode_exact_int64(dimension, shard, tensor, Shape)
    if decoded <= 0L {
      raise InvalidDimension(shard~, tensor~)
    }
    shape.push(decoded)
  }
  shape
}

///|
fn tensor_byte_count(
  shape : ArrayView[Int64],
  dtype : StreamingSafetensorsDTypeSpec,
  shard : Int,
  tensor : Int,
) -> Int64 raise StreamingSafetensorsError {
  let mut elements = 1L
  for dimension in shape {
    if elements > 0x7fffffffffffffffL / dimension {
      raise TensorSizeOverflow(shard~, tensor~)
    }
    elements *= dimension
  }
  if elements > 0x7fffffffffffffffL / dtype.bytes_per_element {
    raise TensorSizeOverflow(shard~, tensor~)
  }
  elements * dtype.bytes_per_element
}

///|
fn valid_tensor_name(name : StringView, maximum : Int) -> Bool {
  !name.is_empty() &&
  name.length() <= maximum &&
  name.all(character => character.is_ascii() && !character.is_ascii_control())
}

///|
fn decode_tensor(
  name : String,
  value : Json,
  shard : Int,
  tensor : Int,
  payload_bytes : Int64,
  specs : ArrayView[StreamingSafetensorsDTypeSpec],
  limits : StreamingSafetensorsLimits,
) -> StreamingSafetensorsTensorMetadata raise StreamingSafetensorsError {
  guard value is Object(fields) else { raise InvalidTensor(shard~, tensor~) }
  for field_name, _ in fields {
    if field_name != "dtype" &&
      field_name != "shape" &&
      field_name != "data_offsets" {
      raise UnexpectedTensorField(shard~, tensor~)
    }
  }
  if !valid_tensor_name(name, limits.maximum_name_bytes) {
    raise InvalidTensorName(shard~, tensor~)
  }
  guard require_tensor_field(fields, "dtype", shard, tensor, DType)
    is String(dtype_tag) else {
    raise InvalidTensorField(shard~, tensor~, DType)
  }
  guard dtype_for_tag(specs, dtype_tag) is Some(dtype) else {
    raise UnknownDType(shard~, tensor~)
  }
  let shape = decode_shape(
    require_tensor_field(fields, "shape", shard, tensor, Shape),
    shard,
    tensor,
    limits,
  )
  guard require_tensor_field(fields, "data_offsets", shard, tensor, DataOffsets)
    is Array(offsets) &&
    offsets.length() == 2 else {
    raise InvalidTensorField(shard~, tensor~, DataOffsets)
  }
  let start = decode_exact_int64(offsets[0], shard, tensor, DataOffsets)
  let end = decode_exact_int64(offsets[1], shard, tensor, DataOffsets)
  if start < 0L || end < start {
    raise InvalidByteRange(shard~, tensor~)
  }
  if end > payload_bytes {
    raise RangeOutsidePayload(shard~, tensor~)
  }
  if end - start != tensor_byte_count(shape, dtype, shard, tensor) {
    raise RangeSizeMismatch(shard~, tensor~)
  }
  {
    name,
    dtype,
    shape: ReadOnlyArray::from_array(shape),
    data_start: start,
    data_end: end,
  }
}

///|
fn validate_document_metadata(
  value : Json,
  shard : Int,
) -> Unit raise StreamingSafetensorsError {
  guard value is Object(entries) else { raise InvalidDocumentMetadata(shard) }
  for _, metadata_value in entries {
    if !(metadata_value is String(_)) {
      raise InvalidDocumentMetadata(shard)
    }
  }
}

///|
fn parse_header(
  header : FixedArray[Byte],
  shard : Int,
  payload_bytes : Int64,
  specs : ArrayView[StreamingSafetensorsDTypeSpec],
  limits : StreamingSafetensorsLimits,
) -> Array[StreamingSafetensorsTensorMetadata] raise StreamingSafetensorsError {
  if !header_has_valid_envelope(header) {
    raise InvalidHeaderEnvelope(shard)
  }
  let text = @utf8.decode(Bytes::from_array(header[:])) catch {
    _ => raise InvalidHeaderEncoding(shard)
  }
  @json_guard.reject_duplicate_object_keys(text, max_depth=4) catch {
    DuplicateObjectKey(depth~, index~) =>
      raise DuplicateHeaderKey(shard~, depth~, index~)
    _ => raise InvalidHeaderJson(shard)
  }
  let root = @json.parse(text) catch { _ => raise InvalidHeaderJson(shard) }
  guard root is Object(fields) else { raise InvalidRoot(shard) }
  let tensors : Array[StreamingSafetensorsTensorMetadata] = []
  for name, value in fields {
    if name == "__metadata__" {
      validate_document_metadata(value, shard)
      continue
    }
    if tensors.length() >= limits.maximum_tensors_per_shard {
      raise TooManyTensors(shard)
    }
    let tensor = decode_tensor(
      name,
      value,
      shard,
      tensors.length(),
      payload_bytes,
      specs,
      limits,
    )
    tensors.push(tensor)
  }
  // JSON key validation already rejects duplicate (including escaped-equivalent)
  // names. Sort an ordinal view for interval validation, retaining the public
  // tensor order and diagnostics in original header coordinates. A checkpoint
  // may concentrate tens of thousands of scalar scales in one auxiliary shard;
  // comparing every pair makes that otherwise small header quadratic.
  validate_tensor_ranges(tensors, shard)
  tensors
}

///|
fn validate_tensor_ranges(
  tensors : Array[StreamingSafetensorsTensorMetadata],
  shard : Int,
) -> Unit raise StreamingSafetensorsError {
  let ordered = tensors.mapi((ordinal, tensor) => (ordinal, tensor))
  ordered.sort_by((a, b) => {
    let by_start = a.1.data_start.compare(b.1.data_start)
    if by_start == 0 {
      a.0.compare(b.0)
    } else {
      by_start
    }
  })
  for index in 1..<ordered.length() {
    let (left_ordinal, left) = ordered[index - 1]
    let (right_ordinal, right) = ordered[index]
    if right.data_start < left.data_end {
      raise OverlappingTensorRanges(
        shard~,
        first=left_ordinal.min(right_ordinal),
        second=left_ordinal.max(right_ordinal),
      )
    }
  }
}
