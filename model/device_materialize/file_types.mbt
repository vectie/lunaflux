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
}

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
} derive(Eq)
