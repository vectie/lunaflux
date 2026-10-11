///|
/// Offline observation only. The caller must retire the bound stream before
/// reading a stage, and before a later producer reuses this shared workspace.
/// Normal prediction execution never calls this method.
pub fn PredictionVariant::capture_block_stage(
  self : Self,
  stage : @joint_diffusion_device.RefinerStage,
) -> Bytes raise @device.DeviceError {
  guard self.block.ready && self.block.workspace is Some(workspace) else {
    raise @device.DeviceError::InvalidArgument(CopyToHost)
  }
  workspace.capture_refiner_stage(stage)
}
