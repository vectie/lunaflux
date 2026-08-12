///|
const SAFETENSORS_PREFIX_BYTES : Int = 8

///|
const MAX_INT_AS_UINT64 : UInt64 = 2147483647UL

///|
fn decode_header_length(file : BytesView) -> UInt64 {
  let mut length = 0UL
  for index in 0..<SAFETENSORS_PREFIX_BYTES {
    length = length | (file[index].to_uint64() << (index * 8))
  }
  length
}

///|
fn header_has_valid_envelope(header : BytesView) -> Bool {
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
/// Decodes a mathematically integral JSON number without lossy conversion.
///
/// Core JSON retains textual representation for integers outside its exact
/// Double envelope; those values take the checked string-to-Int64 path.
/// Exactly integral forms such as `1.0` and `1e0` are accepted, while
/// fractional values and Int64 overflow are rejected.
fn decode_exact_int64(
  value : Json,
  tensor_index : Int,
  field : TensorField,
) -> Int64 raise SafetensorsReaderError {
  guard value is Number(number, repr~) else {
    raise InvalidInteger(tensor_index, field)
  }
  match repr {
    Some(encoded) => {
      let integer : Int64 = @strconv.from_str(encoded) catch {
        _ => raise InvalidInteger(tensor_index, field)
      }
      integer
    }
    None => {
      if number.is_nan() {
        raise InvalidInteger(tensor_index, field)
      }
      let integer = number.to_int64()
      if integer.to_double() != number {
        raise InvalidInteger(tensor_index, field)
      }
      integer
    }
  }
}

///|
fn validate_document_metadata(
  value : Json,
) -> Unit raise SafetensorsReaderError {
  guard value is Object(entries) else { raise InvalidMetadata }
  for _, metadata_value in entries {
    if !(metadata_value is String(_)) {
      raise InvalidMetadata
    }
  }
}

///|
fn require_tensor_field(
  tensor : Map[String, Json],
  name : String,
  tensor_index : Int,
  field : TensorField,
) -> Json raise SafetensorsReaderError {
  match tensor.get(name) {
    Some(value) => value
    None => raise MissingTensorField(tensor_index, field)
  }
}

///|
fn decode_shape(
  value : Json,
  tensor_index : Int,
  limits : @safetensors.HeaderLimits,
) -> Array[Int64] raise SafetensorsReaderError {
  guard value is Array(dimensions) else {
    raise InvalidTensorField(tensor_index, Shape)
  }
  if dimensions.length() > limits.max_rank() {
    raise Validation(InvalidRank(tensor_index))
  }
  let shape : Array[Int64] = []
  for dimension in dimensions {
    shape.push(decode_exact_int64(dimension, tensor_index, Shape))
  }
  shape
}

///|
fn decode_offsets(
  value : Json,
  tensor_index : Int,
) -> (Int64, Int64) raise SafetensorsReaderError {
  guard value is Array(offsets) && offsets.length() == 2 else {
    raise InvalidTensorField(tensor_index, DataOffsets)
  }
  (
    decode_exact_int64(offsets[0], tensor_index, DataOffsets),
    decode_exact_int64(offsets[1], tensor_index, DataOffsets),
  )
}

///|
fn decode_tensor(
  name : String,
  value : Json,
  tensor_index : Int,
  limits : @safetensors.HeaderLimits,
) -> @safetensors.RawTensorMetadata raise SafetensorsReaderError {
  guard value is Object(tensor) else { raise InvalidTensor(tensor_index) }
  for field_name, _ in tensor {
    if field_name != "dtype" &&
      field_name != "shape" &&
      field_name != "data_offsets" {
      raise UnexpectedTensorField(tensor_index)
    }
  }
  let dtype_value = require_tensor_field(tensor, "dtype", tensor_index, DType)
  guard dtype_value is String(dtype) else {
    raise InvalidTensorField(tensor_index, DType)
  }
  let shape = decode_shape(
    require_tensor_field(tensor, "shape", tensor_index, Shape),
    tensor_index,
    limits,
  )
  let (data_start, data_end) = decode_offsets(
    require_tensor_field(tensor, "data_offsets", tensor_index, DataOffsets),
    tensor_index,
  )
  @safetensors.RawTensorMetadata::new(
    name~,
    dtype~,
    shape~,
    data_start~,
    data_end~,
  )
}

///|
fn decode_entries(
  root : Json,
  limits : @safetensors.HeaderLimits,
) -> Array[@safetensors.RawTensorMetadata] raise SafetensorsReaderError {
  guard root is Object(fields) else { raise InvalidRoot }
  let tensors : Array[@safetensors.RawTensorMetadata] = []
  for name, value in fields {
    if name == "__metadata__" {
      validate_document_metadata(value)
      continue
    }
    if tensors.length() >= limits.max_tensor_count() {
      raise Validation(TooManyTensors)
    }
    tensors.push(decode_tensor(name, value, tensors.length(), limits))
  }
  tensors
}

///|
/// Parses and validates an exact safetensors prefix plus header against a
/// separately established payload byte length.
///
/// `prefix_and_header` must contain exactly the eight-byte length prefix and
/// the declared header, with no payload bytes. This form lets a bounded file
/// loader validate metadata before allocating a device destination or reading
/// the model-sized payload into host memory.
pub fn read_header(
  prefix_and_header : BytesView,
  payload_bytes~ : Int64,
  limits~ : @safetensors.HeaderLimits,
) -> @safetensors.SafetensorsMetadata raise SafetensorsReaderError {
  if prefix_and_header.length() < SAFETENSORS_PREFIX_BYTES {
    raise MissingHeaderLength
  }
  if payload_bytes < 0L {
    raise InvalidPayloadLength
  }
  let encoded_length = decode_header_length(prefix_and_header)
  if encoded_length > MAX_INT_AS_UINT64 {
    raise HeaderLengthOverflow
  }
  let header_length = encoded_length.to_int()
  if header_length > limits.max_header_bytes() {
    raise HeaderTooLarge
  }
  let available_after_prefix = prefix_and_header.length() -
    SAFETENSORS_PREFIX_BYTES
  if header_length > available_after_prefix {
    raise TruncatedHeader
  }
  if header_length != available_after_prefix {
    raise HeaderEnvelopeSizeMismatch
  }
  if payload_bytes > limits.max_payload_bytes() {
    raise Validation(RangeOutsidePayload(0))
  }
  let header = prefix_and_header[SAFETENSORS_PREFIX_BYTES:]
  if !header_has_valid_envelope(header) {
    raise InvalidHeaderEnvelope
  }
  let header_text = @utf8.decode(header) catch {
    _ => raise InvalidHeaderEncoding
  }
  @json_guard.reject_duplicate_object_keys(header_text, max_depth=4) catch {
    DuplicateObjectKey(depth~, index~) =>
      raise DuplicateHeaderKey(depth~, index~)
    _ => raise InvalidHeaderJson
  }
  let root = @json.parse(header_text) catch { _ => raise InvalidHeaderJson }
  let entries = decode_entries(root, limits)
  let metadata = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=header_length,
    payload_bytes~,
    limits~,
  ) catch {
    error => raise Validation(error)
  }
  if metadata.materialized_tensor_bytes() != payload_bytes {
    raise PayloadSizeMismatch
  }
  metadata
}

///|
/// Parses and validates metadata from one complete safetensors byte sequence.
///
/// The first eight bytes are decoded as an unsigned little-endian header
/// length. That length and the remaining payload are checked against `limits`
/// before UTF-8 decoding or JSON parsing. The header must begin with `{` and
/// may contain only ASCII spaces after its closing `}`. Every remaining byte is
/// treated as tensor payload, and the validated tensor ranges must account for
/// that payload exactly.
pub fn read_metadata(
  file : BytesView,
  limits~ : @safetensors.HeaderLimits,
) -> @safetensors.SafetensorsMetadata raise SafetensorsReaderError {
  if file.length() < SAFETENSORS_PREFIX_BYTES {
    raise MissingHeaderLength
  }
  let encoded_length = decode_header_length(file)
  if encoded_length > MAX_INT_AS_UINT64 {
    raise HeaderLengthOverflow
  }
  let header_length = encoded_length.to_int()
  if header_length > limits.max_header_bytes() {
    raise HeaderTooLarge
  }
  let available_after_prefix = file.length() - SAFETENSORS_PREFIX_BYTES
  if header_length > available_after_prefix {
    raise TruncatedHeader
  }
  let header_end = SAFETENSORS_PREFIX_BYTES + header_length
  let payload_bytes = (file.length() - header_end).to_int64()
  read_header(file[:header_end], payload_bytes~, limits~)
}
