///|
/// Startup only. Callers must retain every inserted effect owner until close.
pub fn OrderedKernelExecutor::begin_effect_capture(
  self : Self,
) -> Unit raise DeviceError {
  self.raw.begin_effect_capture() catch {
    e => raise error_from_cuda(e)
  }
}

///|
pub fn OrderedKernelExecutor::end_effect_capture(
  self : Self,
) -> Unit raise DeviceError {
  self.raw.end_effect_capture() catch {
    e => raise error_from_cuda(e)
  }
  self.mode = OrderedKernelCaptured
}
