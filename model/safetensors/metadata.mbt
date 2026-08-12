///|
/// Raw, decoded tensor header fields awaiting validation.
///
/// The shape is defensively copied. This type represents decoded metadata, not
/// tensor bytes, and intentionally performs no JSON parsing or materialization.
pub struct RawTensorMetadata {
  name : String
  dtype : String
  shape : ReadOnlyArray[Int64]
  data_start : Int64
  data_end : Int64
} derive(Debug, Eq)

///|
/// Dtypes safe to expose to the v1 BF16 materializer.
pub(all) enum TensorDType {
  BFloat16
} derive(Debug, Eq)

///|
/// Validated tensor metadata with a payload-relative half-open byte range.
pub struct TensorByteRange {
  start : Int64
  end : Int64
} derive(Debug, Eq)

///|
/// Validated tensor metadata with a payload-relative half-open byte range.
pub struct TensorMetadata {
  name : String
  dtype : TensorDType
  shape : ReadOnlyArray[Int64]
  byte_range : TensorByteRange
} derive(Debug, Eq)

///|
/// Validated immutable safetensors header metadata.
pub struct SafetensorsMetadata {
  tensors : ReadOnlyArray[TensorMetadata]
  encoded_header_bytes : Int
  payload_bytes : Int64
  materialized_tensor_bytes : Int64
} derive(Debug, Eq)

///|
/// Creates one decoded tensor entry while defensively copying its shape.
pub fn RawTensorMetadata::new(
  name~ : String,
  dtype~ : String,
  shape~ : ArrayView[Int64],
  data_start~ : Int64,
  data_end~ : Int64,
) -> RawTensorMetadata {
  { name, dtype, shape: ReadOnlyArray::from_array(shape), data_start, data_end }
}

///|
fn valid_tensor_name(name : StringView, max_length : Int) -> Bool {
  !name.is_empty() &&
  name.length() <= max_length &&
  name.all(c => c.is_ascii() && !c.is_ascii_control())
}

///|
fn checked_tensor_bytes(
  shape : ReadOnlyArray[Int64],
  tensor_index : Int,
  max_rank : Int,
) -> Int64 raise SafetensorsError {
  if shape.is_empty() || shape.length() > max_rank {
    raise InvalidRank(tensor_index)
  }
  let max_int64 = 9223372036854775807L
  let mut elements = 1L
  for dimension in shape {
    if dimension <= 0L {
      raise InvalidDimension(tensor_index)
    }
    if elements > max_int64 / dimension {
      raise TensorSizeOverflow(tensor_index)
    }
    elements = elements * dimension
  }
  if elements > max_int64 / 2L {
    raise TensorSizeOverflow(tensor_index)
  }
  elements * 2L
}

///|
fn validate_tensor(
  raw : RawTensorMetadata,
  tensor_index : Int,
  payload_bytes : Int64,
  limits : HeaderLimits,
) -> TensorMetadata raise SafetensorsError {
  if !valid_tensor_name(raw.name, limits.max_name_length()) {
    raise InvalidTensorName(tensor_index)
  }
  if raw.dtype != "BF16" {
    raise InvalidDType(tensor_index)
  }
  let byte_length = checked_tensor_bytes(
    raw.shape,
    tensor_index,
    limits.max_rank(),
  )
  if raw.data_start < 0L || raw.data_end < raw.data_start {
    raise InvalidByteRange(tensor_index)
  }
  if raw.data_end > payload_bytes {
    raise RangeOutsidePayload(tensor_index)
  }
  if raw.data_end - raw.data_start != byte_length {
    raise RangeSizeMismatch(tensor_index)
  }
  {
    name: raw.name,
    dtype: BFloat16,
    shape: raw.shape,
    byte_range: { start: raw.data_start, end: raw.data_end },
  }
}

///|
fn ranges_overlap(left : TensorMetadata, right : TensorMetadata) -> Bool {
  left.byte_range.start < right.byte_range.end &&
  right.byte_range.start < left.byte_range.end
}

///|
/// Validates bounded decoded header metadata without reading or materializing
/// the tensor payload.
pub fn validate_metadata(
  entries : ArrayView[RawTensorMetadata],
  encoded_header_bytes~ : Int,
  payload_bytes~ : Int64,
  limits~ : HeaderLimits,
) -> SafetensorsMetadata raise SafetensorsError {
  if encoded_header_bytes < 0 ||
    encoded_header_bytes > limits.max_header_bytes() {
    raise HeaderTooLarge
  }
  if entries.length() > limits.max_tensor_count() {
    raise TooManyTensors
  }
  if payload_bytes < 0L || payload_bytes > limits.max_payload_bytes() {
    raise RangeOutsidePayload(0)
  }
  let validated : Array[TensorMetadata] = []
  let mut total = 0L
  for index, raw in entries {
    let tensor = validate_tensor(raw, index, payload_bytes, limits)
    for previous_index, previous in validated {
      if previous.name == tensor.name {
        raise DuplicateTensor(previous_index, index)
      }
      if ranges_overlap(previous, tensor) {
        raise OverlappingTensors(previous_index, index)
      }
    }
    let budget = limits.max_materialized_tensor_bytes()
    if tensor.byte_range.byte_length() > budget - total {
      raise MaterializedSizeExceeded
    }
    total = total + tensor.byte_range.byte_length()
    validated.push(tensor)
  }
  {
    tensors: ReadOnlyArray::from_array(validated),
    encoded_header_bytes,
    payload_bytes,
    materialized_tensor_bytes: total,
  }
}

///|
pub fn TensorMetadata::name(self : TensorMetadata) -> String {
  self.name
}

///|
pub fn TensorMetadata::dtype(self : TensorMetadata) -> TensorDType {
  self.dtype
}

///|
pub fn TensorMetadata::shape(self : TensorMetadata) -> ReadOnlyArray[Int64] {
  self.shape
}

///|
pub fn TensorMetadata::data_start(self : TensorMetadata) -> Int64 {
  self.byte_range.start
}

///|
pub fn TensorMetadata::data_end(self : TensorMetadata) -> Int64 {
  self.byte_range.end
}

///|
pub fn TensorMetadata::byte_length(self : TensorMetadata) -> Int64 {
  self.byte_range.byte_length()
}

///|
pub fn TensorMetadata::byte_range(self : TensorMetadata) -> TensorByteRange {
  self.byte_range
}

///|
pub fn TensorByteRange::start(self : TensorByteRange) -> Int64 {
  self.start
}

///|
pub fn TensorByteRange::end(self : TensorByteRange) -> Int64 {
  self.end
}

///|
pub fn TensorByteRange::byte_length(self : TensorByteRange) -> Int64 {
  self.end - self.start
}

///|
pub fn SafetensorsMetadata::tensor_count(self : SafetensorsMetadata) -> Int {
  self.tensors.length()
}

///|
pub fn SafetensorsMetadata::tensor(
  self : SafetensorsMetadata,
  index : Int,
) -> TensorMetadata? {
  self.tensors.get(index)
}

///|
pub fn SafetensorsMetadata::find_tensor(
  self : SafetensorsMetadata,
  name : StringView,
) -> TensorMetadata? {
  for tensor in self.tensors {
    if tensor.name[:] == name {
      return Some(tensor)
    }
  }
  None
}

///|
pub fn SafetensorsMetadata::encoded_header_bytes(
  self : SafetensorsMetadata,
) -> Int {
  self.encoded_header_bytes
}

///|
pub fn SafetensorsMetadata::payload_bytes(self : SafetensorsMetadata) -> Int64 {
  self.payload_bytes
}

///|
pub fn SafetensorsMetadata::materialized_tensor_bytes(
  self : SafetensorsMetadata,
) -> Int64 {
  self.materialized_tensor_bytes
}
