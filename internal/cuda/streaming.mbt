///|
type StreamingPool

///|
pub struct StreamingPreparation {
  priv raw : StreamingPool
  priv status : Int
}

///|
#borrow(context, allocation, status)
extern "c" fn raw_streaming_create(
  context : Context,
  allocation : Allocation,
  bytes : Int64,
  lanes : Int,
  regions : Int,
  status : Ref[Int],
) -> StreamingPool = "lunaflux_cuda_streaming_create"

///|
#borrow(pool)
extern "c" fn raw_streaming_close(pool : StreamingPool) -> Int = "lunaflux_cuda_streaming_close"

///|
#borrow(pool)
extern "c" fn raw_streaming_enqueue(
  pool : StreamingPool,
  lane : Int,
  host_offset : Int64,
  device_offset : Int64,
  bytes : Int64,
  direction : Int,
) -> Int = "lunaflux_cuda_streaming_enqueue"

///|
#borrow(pool)
extern "c" fn raw_streaming_record(pool : StreamingPool, lane : Int) -> Int = "lunaflux_cuda_streaming_record"

///|
#borrow(pool)
extern "c" fn raw_streaming_poll(pool : StreamingPool, lane : Int) -> Int = "lunaflux_cuda_streaming_poll"

///|
#borrow(pool)
extern "c" fn raw_streaming_drain(pool : StreamingPool, lane : Int) -> Int = "lunaflux_cuda_streaming_drain"

///|
#borrow(pool)
extern "c" fn raw_streaming_reset(pool : StreamingPool, lane : Int) -> Int = "lunaflux_cuda_streaming_reset"

///|
#borrow(pool, buffer)
extern "c" fn raw_streaming_host_copy(
  pool : StreamingPool,
  buffer : FixedArray[Byte],
  host_offset : Int64,
  offset : Int,
  bytes : Int,
  direction : Int,
) -> Int = "lunaflux_cuda_streaming_host_copy"

///|
/// Always returns cleanup authority, including partially created resources.
pub fn Context::prepare_streaming(
  self : Context,
  allocation : Allocation,
  host_bytes : Int64,
  lanes : Int,
  max_regions : Int,
) -> StreamingPreparation {
  let status = Ref(0)
  let raw = raw_streaming_create(
    self, allocation, host_bytes, lanes, max_regions, status,
  )
  { raw, status: status.val, }
}

///|
pub fn StreamingPreparation::is_ready(self : StreamingPreparation) -> Bool {
  self.status == 0
}

///|
pub fn StreamingPreparation::check_prepared(
  self : StreamingPreparation,
) -> Unit raise CudaError {
  check_status(self.status, StreamingTransfer)
}

///|
pub fn StreamingPreparation::enqueue(
  self : StreamingPreparation,
  lane : Int,
  host_offset : Int64,
  device_offset : Int64,
  bytes : Int64,
  to_device : Bool,
) -> Unit raise CudaError {
  check_status(
    raw_streaming_enqueue(
      self.raw,
      lane,
      host_offset,
      device_offset,
      bytes,
      if to_device {
        1
      } else {
        0
      },
    ),
    StreamingTransfer,
  )
}

///|
pub fn StreamingPreparation::record(
  self : StreamingPreparation,
  lane : Int,
) -> Unit raise CudaError {
  check_status(raw_streaming_record(self.raw, lane), StreamingTransfer)
}

///|
pub fn StreamingPreparation::poll(
  self : StreamingPreparation,
  lane : Int,
) -> Bool raise CudaError {
  match raw_streaming_poll(self.raw, lane) {
    0 => false
    1 => true
    status => {
      check_status(status, StreamingTransfer)
      false
    }
  }
}

///|
pub fn StreamingPreparation::drain(
  self : StreamingPreparation,
  lane : Int,
) -> Unit raise CudaError {
  check_status(raw_streaming_drain(self.raw, lane), StreamingTransfer)
}

///|
pub fn StreamingPreparation::reset(
  self : StreamingPreparation,
  lane : Int,
) -> Unit raise CudaError {
  check_status(raw_streaming_reset(self.raw, lane), StreamingTransfer)
}

///|
pub fn StreamingPreparation::close(
  self : StreamingPreparation,
) -> Unit raise CudaError {
  check_status(raw_streaming_close(self.raw), StreamingTransfer)
}

///|
pub fn StreamingPreparation::host_copy(
  self : StreamingPreparation,
  buffer : FixedArray[Byte],
  host_offset : Int64,
  offset : Int,
  bytes : Int,
  write : Bool,
) -> Unit raise CudaError {
  check_status(
    raw_streaming_host_copy(
      self.raw,
      buffer,
      host_offset,
      offset,
      bytes,
      if write {
        1
      } else {
        0
      },
    ),
    StreamingTransfer,
  )
}
