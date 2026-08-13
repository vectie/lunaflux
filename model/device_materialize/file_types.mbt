///|
const MAX_TRANSFER_CHUNK_BYTES : Int = 16 * 1024 * 1024

///|
/// Startup bounds for streaming one safetensors file into device memory.
///
/// The header has its own independent bound in `safetensors`. Payload I/O and
/// host-to-device copies are split into chunks no larger than
/// `transfer_chunk_bytes`; the loader never owns a complete payload snapshot.
pub struct DeviceWeightFileLimits {
  max_file_bytes : Int64
  transfer_chunk_bytes : Int
  safetensors : @safetensors.HeaderLimits
} derive(Debug, Eq)

///|
pub fn DeviceWeightFileLimits::new(
  max_file_bytes~ : Int64,
  transfer_chunk_bytes~ : Int,
  safetensors~ : @safetensors.HeaderLimits,
) -> DeviceWeightFileLimits raise DeviceWeightError {
  if max_file_bytes <= 8L {
    raise InvalidFileLimit(FileBytes)
  }
  if transfer_chunk_bytes <= 0 ||
    transfer_chunk_bytes > MAX_TRANSFER_CHUNK_BYTES {
    raise InvalidFileLimit(TransferChunkBytes)
  }
  { max_file_bytes, transfer_chunk_bytes, safetensors }
}

///|
pub fn DeviceWeightFileLimits::max_file_bytes(
  self : DeviceWeightFileLimits,
) -> Int64 {
  self.max_file_bytes
}

///|
pub fn DeviceWeightFileLimits::transfer_chunk_bytes(
  self : DeviceWeightFileLimits,
) -> Int {
  self.transfer_chunk_bytes
}

///|
pub fn DeviceWeightFileLimits::safetensors(
  self : DeviceWeightFileLimits,
) -> @safetensors.HeaderLimits {
  self.safetensors
}

///|
priv struct StreamingTransferRegion {
  reference : @plan.TensorRef
  source_offset_bytes : Int64
  destination_offset_bytes : Int64
  byte_length : Int64
} derive(Eq)

///|
priv struct PreparedStreamingLoad {
  source_stamp : SourceStamp
  payload_offset_bytes : Int64
  layout : DeviceWeightLayout
  regions : ReadOnlyArray[StreamingTransferRegion]
}

///|
priv struct SourceStamp {
  file_bytes : Int64
  modified_seconds : Int64
  modified_nanoseconds : Int
  changed_seconds : Int64
  changed_nanoseconds : Int
} derive(Eq)

///|
/// Immutable evidence from a complete read-only admission of one weight file.
///
/// The source locator and filesystem metadata are deliberately not exposed or
/// retained as authority. A later load reopens and completely re-hashes the
/// caller-supplied locator before allocating device memory.
pub struct DeviceWeightFileInspection {
  priv model : @spec.LlamaModelMetadata
  priv limits : DeviceWeightFileLimits
  priv alignment_bytes : Int64
  priv max_arena_bytes : Int64
  priv payload_offset_bytes : Int64
  priv layout : DeviceWeightLayout
  priv regions : ReadOnlyArray[StreamingTransferRegion]
}

///|
pub fn DeviceWeightFileInspection::content_digest(
  self : DeviceWeightFileInspection,
) -> @spec.ContentDigest {
  self.layout.identity().content()
}

///|
pub fn DeviceWeightFileInspection::layout(
  self : DeviceWeightFileInspection,
) -> DeviceWeightLayout {
  self.layout
}
