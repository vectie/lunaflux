///|
#borrow(executor)
extern "c" fn raw_capture_begin(executor : OrderedKernelExecutor) -> Int = "lunaflux_cuda_ordered_capture_begin"

///|
#borrow(executor)
extern "c" fn raw_capture_end(executor : OrderedKernelExecutor) -> Int = "lunaflux_cuda_ordered_capture_end"

///|
pub fn OrderedKernelExecutor::begin_effect_capture(
  self : Self,
) -> Unit raise CudaError {
  check_status(raw_capture_begin(self), OrderedExecutorCreate)
}

///|
pub fn OrderedKernelExecutor::end_effect_capture(
  self : Self,
) -> Unit raise CudaError {
  check_status(raw_capture_end(self), OrderedExecutorCreate)
}
