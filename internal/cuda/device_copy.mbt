///|
type DeviceCopyBatch

///|
#borrow(context, stream, sources, destinations, source_offsets, destination_offsets, sizes, status)
extern "c" fn raw_device_copy_create(
  context : Context,
  stream : Stream,
  sources : FixedArray[Allocation],
  destinations : FixedArray[Allocation],
  source_offsets : FixedArray[Int64],
  destination_offsets : FixedArray[Int64],
  sizes : FixedArray[Int64],
  status : Ref[Int],
) -> DeviceCopyBatch = "lunaflux_cuda_device_copy_create"

///|
#borrow(copy)
extern "c" fn raw_device_copy_submit(
  copy : DeviceCopyBatch,
  reverse : Bool,
) -> Int = "lunaflux_cuda_device_copy_submit"

///|
#borrow(copy)
extern "c" fn raw_device_copy_retire(
  copy : DeviceCopyBatch,
  wait : Bool,
) -> Int = "lunaflux_cuda_device_copy_retire"

///|
#borrow(copy)
extern "c" fn raw_device_copy_close(copy : DeviceCopyBatch) -> Int = "lunaflux_cuda_device_copy_close"

///|
pub fn prepare_device_copy_batch(
  context : Context,
  stream : Stream,
  sources : FixedArray[Allocation],
  destinations : FixedArray[Allocation],
  source_offsets : FixedArray[Int64],
  destination_offsets : FixedArray[Int64],
  sizes : FixedArray[Int64],
) -> DeviceCopyBatch raise CudaError {
  let status = Ref(0)
  let copy = raw_device_copy_create(
    context, stream, sources, destinations, source_offsets, destination_offsets,
    sizes, status,
  )
  if status.val != 0 {
    check_status(raw_device_copy_close(copy), DeviceCopyClose)
    check_status(status.val, DeviceCopyCreate)
  }
  copy
}

///|
pub fn DeviceCopyBatch::submit(
  self : Self,
  reverse~ : Bool,
) -> Unit raise CudaError {
  check_status(raw_device_copy_submit(self, reverse), DeviceCopySubmit)
}

///|
pub fn DeviceCopyBatch::poll(self : Self) -> Bool raise CudaError {
  match raw_device_copy_retire(self, false) {
    0 => false
    1 => true
    status => {
      check_status(status, DeviceCopyPoll)
      false
    }
  }
}

///|
pub fn DeviceCopyBatch::wait(self : Self) -> Unit raise CudaError {
  let status = raw_device_copy_retire(self, true)
  if status != 1 {
    check_status(status, DeviceCopyWait)
  }
}

///|
pub fn DeviceCopyBatch::close(self : Self) -> Unit raise CudaError {
  check_status(raw_device_copy_close(self), DeviceCopyClose)
}
