///|
/// Additional rollback budget, separate from the resident predictor footprint.
/// Includes every committed-main ring/frontier and the explicit 16-byte RNG.
pub fn DeepSeekPredictionExecution::state_backup_bytes(self : Self) -> Int64 {
  self.predictor.persistent_state_bytes() + 16L
}

///|
/// Enumerate only at startup while idle. Backup/restore must use the same stream
/// as predictor queues; retire model work before rolling back or committing.
pub fn DeepSeekPredictionExecution::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared && !self.leased else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  let buffers = self.predictor.persistent_state()
  buffers.push(self.results[2])
  buffers
}

///|
/// Prepare once before execution; no host staging or allocation on begin/undo.
/// Close the snapshot before the predictor whose allocations it leases.
pub fn DeepSeekPredictionExecution::prepare_state_snapshot(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  maximum_backup_bytes~ : Int64,
) -> @snapshot.DeviceStateSnapshot raise {
  @snapshot.prepare_device_state_snapshot(
    context,
    stream,
    self.persistent_state(),
    maximum_backup_bytes~,
  )
}
