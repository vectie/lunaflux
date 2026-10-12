///|
/// Offline ports from the executed attention/indexer compressor. Snapshot or
/// discard all persistent state when aborting; history alone is not rollback.
pub fn DeepSeekCompressorPool::replay_buffers(
  self : Self,
) -> Array[(@precision.NumericBoundary, @device.Allocation)] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  self.frame.replay_buffers()
}
