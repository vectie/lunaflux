///|
/// Startup-only exact base-state backup budget, available before GPU allocation.
pub fn DeepSeekDecoderStage::state_backup_bytes(self : Self) -> Int64 {
  self.decoder.layers.fold(init=0L, (total, layer) => {
    total + layer.block.attention.persistent_state_bytes()
  })
}

///|
/// Cancellation barrier without releasing state borrowed by snapshot owners.
/// Activation/prediction borrowers must retire first. No successful request
/// publication is implied; this owner is terminal and only close may follow.
pub fn DeepSeekDecoderStage::drain(self : Self) -> Unit raise {
  guard !self.activation_leased && !self.prediction_leased else {
    raise @device.DeviceError::Busy(OrderedExecutorAbort)
  }
  if self.state is Closed {
    return
  }
  self.state = Failed
  if self.decoder.queue is Some(queue) {
    queue.abort()
  }
}

///|
/// Declare actual mutable base state, not weights or scratch. Enumeration is
/// startup-only and includes learned index caches and incomplete pooling groups.
pub fn DeepSeekDecoderStage::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared &&
    !self.activation_leased &&
    !self.prediction_leased else {
    raise @device.DeviceError::Busy(DeviceCopyCreate)
  }
  let state = []
  for layer in self.decoder.layers {
    state.append(layer.block.attention.persistent_state())
  }
  state
}

///|
/// Optional verification state owner. Budget its backup bytes in addition to
/// required_device_bytes; close the transaction before closing decoder state.
/// Publication/host request frontiers remain the outer scheduler transaction.
pub fn DeepSeekDecoderStage::prepare_state_snapshot(
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
