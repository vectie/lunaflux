///|
/// Prepare checkpoint-backed layers for a model-owned queue. Frame preparation
/// uses the same weights and stages as standalone execution, without creating
/// a layer queue. The model must close its queue before closing these owners.
pub fn GlmRecurrentLayer::prepare_frame(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  self.prepare_layer(
    context,
    aot,
    None,
    checkpoint,
    root,
    maximum_device_bytes~,
    chunk_bytes~,
    limits~,
    counts~,
    sequence_offsets~,
    sequence_slots~,
    reset_flags~,
    hidden~,
    output~,
  )
}

///|
pub fn GlmRecurrentLayer::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.program.launches()
}

///|
pub fn GlmSparseLayer::prepare_frame(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise {
  self.prepare_layer(
    context,
    aot,
    None,
    checkpoint,
    root,
    maximum_device_bytes~,
    chunk_bytes~,
    limits~,
    counts~,
    reset_flag~,
    hidden~,
    output~,
  )
}

///|
pub fn GlmSparseLayer::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}
