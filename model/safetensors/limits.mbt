///|
/// Explicit allocation and payload envelope for one decoded safetensors header.
pub struct HeaderLimits {
  max_header_bytes : Int
  max_tensor_count : Int
  max_rank : Int
  max_name_length : Int
  max_payload_bytes : Int64
  max_materialized_tensor_bytes : Int64
} derive(Debug, Eq)

///|
/// Constructs a bounded validation envelope.
pub fn HeaderLimits::new(
  max_header_bytes~ : Int,
  max_tensor_count~ : Int,
  max_rank~ : Int,
  max_name_length~ : Int,
  max_payload_bytes~ : Int64,
  max_materialized_tensor_bytes~ : Int64,
) -> HeaderLimits raise SafetensorsError {
  if max_header_bytes <= 0 {
    raise InvalidLimit(HeaderBytes)
  }
  if max_tensor_count <= 0 {
    raise InvalidLimit(TensorCount)
  }
  if max_rank <= 0 {
    raise InvalidLimit(TensorRank)
  }
  if max_name_length <= 0 {
    raise InvalidLimit(TensorNameLength)
  }
  if max_payload_bytes <= 0L {
    raise InvalidLimit(PayloadBytes)
  }
  if max_materialized_tensor_bytes <= 0L ||
    max_materialized_tensor_bytes > max_payload_bytes {
    raise InvalidLimit(MaterializedTensorBytes)
  }
  {
    max_header_bytes,
    max_tensor_count,
    max_rank,
    max_name_length,
    max_payload_bytes,
    max_materialized_tensor_bytes,
  }
}

///|
pub fn HeaderLimits::max_header_bytes(self : HeaderLimits) -> Int {
  self.max_header_bytes
}

///|
pub fn HeaderLimits::max_tensor_count(self : HeaderLimits) -> Int {
  self.max_tensor_count
}

///|
pub fn HeaderLimits::max_rank(self : HeaderLimits) -> Int {
  self.max_rank
}

///|
pub fn HeaderLimits::max_name_length(self : HeaderLimits) -> Int {
  self.max_name_length
}

///|
pub fn HeaderLimits::max_payload_bytes(self : HeaderLimits) -> Int64 {
  self.max_payload_bytes
}

///|
pub fn HeaderLimits::max_materialized_tensor_bytes(
  self : HeaderLimits,
) -> Int64 {
  self.max_materialized_tensor_bytes
}
