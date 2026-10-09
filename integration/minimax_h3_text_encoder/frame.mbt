///|
/// Synchronous pipeline adapter. One enqueue batch and one event wait, never a
/// polling spin or host wait between kernels. Failed requests cannot replay.
pub fn EncoderRequest::execute(self : Self) -> Unit raise @device.DeviceError {
  self.submit()
  self.wait()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for EncoderRequest with fn execute(
  self,
) {
  self.execute()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for EncoderRequest with fn drain(
  self,
) {
  self.cancel()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for EncoderRequest with fn release(
  self,
) {
  self.close()
}
