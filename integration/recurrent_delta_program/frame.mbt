///|
/// Prepare a complete branch for a caller-owned model queue. Its launch view
/// borrows this program's functions, intermediates and request state. Close the
/// caller's queue before closing this frame; abort cannot drain that queue.
pub fn RecurrentDeltaProgram::prepare_block_frame(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  weights : RecurrentBlockWeights,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
  prefix? : ArrayView[@device.OrderedKernelLaunch] = [],
  suffix? : ArrayView[@device.OrderedKernelLaunch] = [],
) -> Unit raise @device.DeviceError {
  guard self.block is Some(_) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.prepare_program(
    context,
    aot,
    None,
    maximum_state_bytes~,
    maximum_workspace_bytes~,
    counts~,
    sequence_offsets~,
    sequence_slots~,
    reset_flags~,
    query=hidden,
    key=hidden,
    value=hidden,
    log_decay=hidden,
    beta=hidden,
    output~,
    hidden=Some(hidden),
    block_weights=Some(weights),
    convolution_weights=Some(
      FixedArray::from_array([
        weights.query_conv,
        weights.key_conv,
        weights.value_conv,
      ]),
    ),
    prefix~,
    suffix~,
  )
}

///|
pub fn RecurrentDeltaProgram::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}
