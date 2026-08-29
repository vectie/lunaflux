///|
/// Startup bounds for a complete safetensors materialization operation.
pub struct MaterializationLimits {
  priv max_header_bytes : Int
  priv max_tensor_count : Int
  priv max_payload_bytes : Int64
  priv max_host_bytes : Int64
} derive(Debug, Eq)

///|
/// Validated lowercase SHA-256 checksum for one tensor's exact payload bytes.
pub struct TensorChecksum {
  priv hex : String
} derive(Debug, Eq)

///|
/// Immutable zero-copy view delivered to a final-destination visitor.
///
/// `bytes` remains backed by the immutable complete-file input. A visitor must
/// copy it during the callback if its destination does not retain MoonBit
/// values after the call returns.
pub struct TensorChunk {
  priv reference : @plan.TensorRef
  priv tensor : @safetensors.TensorMetadata
  priv file_offset_bytes : Int64
  priv bytes : BytesView
  priv checksum : TensorChecksum
} derive(Debug, Eq)

///|
/// Immutable evidence that one tensor was validated and delivered.
pub struct TensorReceipt {
  priv reference : @plan.TensorRef
  priv byte_length : Int64
  priv checksum : TensorChecksum
} derive(Debug, Eq)

///|
/// Ordered evidence for one complete visitor materialization.
pub struct MaterializationReport {
  priv identity : @spec.ModelIdentity
  priv tensors : ReadOnlyArray[TensorReceipt]
  priv total_bytes : Int64
} derive(Debug, Eq)

///|
/// One independently owned tensor copy for the bounded reference executor.
pub struct HostTensor {
  priv reference : @plan.TensorRef
  priv tensor : @safetensors.TensorMetadata
  priv bytes : Bytes
  priv checksum : TensorChecksum
} derive(Debug, Eq)

///|
/// Bounded host-owned weights in semantic `TensorRef` order.
pub struct HostMaterialization {
  priv identity : @spec.ModelIdentity
  priv tensors : ReadOnlyArray[HostTensor]
  priv total_bytes : Int64
} derive(Debug, Eq)

///|
/// Constructs explicit startup materialization limits.
pub fn MaterializationLimits::new(
  max_header_bytes~ : Int,
  max_tensor_count~ : Int,
  max_payload_bytes~ : Int64,
  max_host_bytes~ : Int64,
) -> MaterializationLimits raise MaterializationError {
  if max_header_bytes <= 0 {
    raise InvalidLimit(HeaderBytes)
  }
  if max_tensor_count <= 0 {
    raise InvalidLimit(TensorCount)
  }
  if max_payload_bytes <= 0L {
    raise InvalidLimit(PayloadBytes)
  }
  if max_host_bytes <= 0L || max_host_bytes > max_payload_bytes {
    raise InvalidLimit(HostBytes)
  }
  { max_header_bytes, max_tensor_count, max_payload_bytes, max_host_bytes }
}

///|
pub fn MaterializationLimits::max_header_bytes(
  self : MaterializationLimits,
) -> Int {
  self.max_header_bytes
}

///|
pub fn MaterializationLimits::max_tensor_count(
  self : MaterializationLimits,
) -> Int {
  self.max_tensor_count
}

///|
pub fn MaterializationLimits::max_payload_bytes(
  self : MaterializationLimits,
) -> Int64 {
  self.max_payload_bytes
}

///|
pub fn MaterializationLimits::max_host_bytes(
  self : MaterializationLimits,
) -> Int64 {
  self.max_host_bytes
}

///|
pub fn TensorChecksum::as_hex(self : TensorChecksum) -> String {
  self.hex
}

///|
pub fn TensorChunk::reference(self : TensorChunk) -> @plan.TensorRef {
  self.reference
}

///|
pub fn TensorChunk::tensor(self : TensorChunk) -> @safetensors.TensorMetadata {
  self.tensor
}

///|
/// Returns the exact byte offset of this tensor in the complete immutable
/// safetensors file supplied to `visit`.
///
/// Device destinations can use this with the original `Bytes` value to issue
/// one direct host-to-device copy without first owning the tensor view.
pub fn TensorChunk::file_offset_bytes(self : TensorChunk) -> Int64 {
  self.file_offset_bytes
}

///|
pub fn TensorChunk::bytes(self : TensorChunk) -> BytesView {
  self.bytes
}

///|
pub fn TensorChunk::checksum(self : TensorChunk) -> TensorChecksum {
  self.checksum
}

///|
pub fn TensorReceipt::reference(self : TensorReceipt) -> @plan.TensorRef {
  self.reference
}

///|
pub fn TensorReceipt::byte_length(self : TensorReceipt) -> Int64 {
  self.byte_length
}

///|
pub fn TensorReceipt::checksum(self : TensorReceipt) -> TensorChecksum {
  self.checksum
}

///|
pub fn MaterializationReport::length(self : MaterializationReport) -> Int {
  self.tensors.length()
}

///|
pub fn MaterializationReport::identity(
  self : MaterializationReport,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn MaterializationReport::receipt(
  self : MaterializationReport,
  reference : @plan.TensorRef,
) -> TensorReceipt? {
  self.tensors.get(reference.as_int())
}

///|
pub fn MaterializationReport::total_bytes(
  self : MaterializationReport,
) -> Int64 {
  self.total_bytes
}

///|
pub fn HostTensor::reference(self : HostTensor) -> @plan.TensorRef {
  self.reference
}

///|
pub fn HostTensor::tensor(self : HostTensor) -> @safetensors.TensorMetadata {
  self.tensor
}

///|
pub fn HostTensor::bytes(self : HostTensor) -> Bytes {
  self.bytes
}

///|
pub fn HostTensor::checksum(self : HostTensor) -> TensorChecksum {
  self.checksum
}

///|
pub fn HostMaterialization::length(self : HostMaterialization) -> Int {
  self.tensors.length()
}

///|
pub fn HostMaterialization::identity(
  self : HostMaterialization,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn HostMaterialization::tensor(
  self : HostMaterialization,
  reference : @plan.TensorRef,
) -> HostTensor? {
  self.tensors.get(reference.as_int())
}

///|
pub fn HostMaterialization::total_bytes(self : HostMaterialization) -> Int64 {
  self.total_bytes
}
