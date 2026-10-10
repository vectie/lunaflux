///|
/// Prepared device-only copy effect. Regions and resource lifetimes are resolved
/// once. Submission does no allocation, payload hashing or host staging.
pub struct DeviceCopyBatch {
  priv raw : @cuda.DeviceCopyBatch
}

///|
pub struct DeviceCopyRegion {
  priv source : Allocation
  priv destination : Allocation
  priv source_offset : Int64
  priv destination_offset : Int64
  priv byte_count : Int64
}

///|
pub fn DeviceCopyRegion::new(
  source : Allocation,
  destination : Allocation,
  source_offset? : Int64 = 0L,
  destination_offset? : Int64 = 0L,
  byte_count~ : Int64,
) -> Self raise DeviceError {
  guard source_offset >= 0L &&
    destination_offset >= 0L &&
    byte_count > 0L &&
    source_offset <= source.byte_count &&
    byte_count <= source.byte_count - source_offset &&
    destination_offset <= destination.byte_count &&
    byte_count <= destination.byte_count - destination_offset else {
    raise InvalidArgument(DeviceCopyCreate)
  }
  { source, destination, source_offset, destination_offset, byte_count, }
}

///|
/// Retains all regions and the stream until close. Destinations must be disjoint
/// from each other and all sources, including for reverse-direction restore.
pub fn Context::prepare_device_copy(
  self : Context,
  stream : Stream,
  regions : ArrayView[DeviceCopyRegion],
) -> DeviceCopyBatch raise DeviceError {
  let raw = @cuda.prepare_device_copy_batch(
    self.raw,
    stream.raw,
    FixedArray::makei(regions.length(), i => regions[i].source.raw),
    FixedArray::makei(regions.length(), i => regions[i].destination.raw),
    FixedArray::makei(regions.length(), i => regions[i].source_offset),
    FixedArray::makei(regions.length(), i => regions[i].destination_offset),
    FixedArray::makei(regions.length(), i => regions[i].byte_count),
  ) catch {
    error => raise error_from_cuda(error)
  }
  { raw, }
}

///|
pub fn DeviceCopyBatch::submit(
  self : Self,
  reverse? : Bool = false,
) -> Unit raise DeviceError {
  self.raw.submit(reverse~) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn DeviceCopyBatch::poll_completion(self : Self) -> Bool raise DeviceError {
  self.raw.poll() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Also drains a submission failure before any operand can be released.
pub fn DeviceCopyBatch::wait_completion(self : Self) -> Unit raise DeviceError {
  self.raw.wait() catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn DeviceCopyBatch::close(self : Self) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}
