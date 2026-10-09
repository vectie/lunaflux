///|
/// Startup-owned pinned storage and bounded asynchronous copy lanes. Native
/// resources and the bound device allocation stay leased until explicit close.
/// The owner is worker-confined and must be retained even when preparation fails.
pub struct TransferPool {
  priv raw : @cuda.StreamingPreparation
}

///|
/// Live startup observation; never used on token-step dispatch.
pub fn Context::has_separate_host_memory(
  self : Context,
) -> Bool raise DeviceError {
  self.raw.separate_host_memory(self.ordinal()) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn Context::prepare_transfer_pool(
  self : Context,
  allocation : Allocation,
  host_bytes~ : Int64,
  lanes~ : Int,
  max_regions_per_lane~ : Int,
) -> TransferPool {
  {
    raw: self.raw.prepare_streaming(
      allocation.raw,
      host_bytes,
      lanes,
      max_regions_per_lane,
    ),
  }
}

///|
/// Reports the startup result only; close still owns cleanup after failure.
pub fn TransferPool::is_prepared(self : TransferPool) -> Bool {
  self.raw.is_ready()
}

///|
pub fn TransferPool::check_prepared(
  self : TransferPool,
) -> Unit raise DeviceError {
  self.raw.check_prepared() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Enqueue one region in a bounded batch. No managed buffer is borrowed by DMA.
/// On failure drain the lane before resetting or releasing its reservations.
pub fn TransferPool::enqueue(
  self : TransferPool,
  lane~ : Int,
  host_offset~ : Int64,
  device_offset~ : Int64,
  bytes~ : Int64,
  to_device~ : Bool,
) -> Unit raise DeviceError {
  self.raw.enqueue(lane, host_offset, device_offset, bytes, to_device) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn TransferPool::record(
  self : TransferPool,
  lane : Int,
) -> Unit raise DeviceError {
  self.raw.record(lane) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn TransferPool::poll(
  self : TransferPool,
  lane : Int,
) -> Bool raise DeviceError {
  self.raw.poll(lane) catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Fault/drain path only. Normal progress uses nonblocking poll.
pub fn TransferPool::drain(
  self : TransferPool,
  lane : Int,
) -> Unit raise DeviceError {
  self.raw.drain(lane) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn TransferPool::reset(
  self : TransferPool,
  lane : Int,
) -> Unit raise DeviceError {
  self.raw.reset(lane) catch {
    error => raise error_from_cuda(error)
  }
}

///|
pub fn TransferPool::close(self : TransferPool) -> Unit raise DeviceError {
  self.raw.close() catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Synchronous bounded staging into owned pinned storage, for startup loading.
pub fn TransferPool::write_host(
  self : TransferPool,
  source : FixedArray[Byte],
  host_offset~ : Int64,
  source_offset~ : Int,
  bytes~ : Int,
) -> Unit raise DeviceError {
  self.raw.host_copy(source, host_offset, source_offset, bytes, true) catch {
    error => raise error_from_cuda(error)
  }
}

///|
/// Explicit diagnostic/startup read; never implicit token-step observation.
pub fn TransferPool::read_host(
  self : TransferPool,
  destination : FixedArray[Byte],
  host_offset~ : Int64,
  destination_offset~ : Int,
  bytes~ : Int,
) -> Unit raise DeviceError {
  self.raw.host_copy(destination, host_offset, destination_offset, bytes, false) catch {
    error => raise error_from_cuda(error)
  }
}
