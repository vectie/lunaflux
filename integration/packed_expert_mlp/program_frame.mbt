///|
/// Prepare a complete local MoE branch without creating a second execution
/// queue. The caller borrows launches into its larger ordered program and must
/// close that queue first. Partitioned expert banks continue to use the
/// collective-aware prepare_routed path; this composition must own every expert.
pub fn MoeProgram::prepare_routed_frame(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias? : @device.Allocation,
  token_ids? : @device.Allocation,
  token_table? : @device.Allocation,
  token_table_offset? : Int64 = 0L,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  self.check_fresh(maximum_workspace_bytes)
  guard self.routed.plan().local_experts().length() ==
    self.routed.plan().geometry().3 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (indices, weights, prefix) = self.bind_router(
    context,
    aot_module,
    counts~,
    input~,
    router_weight~,
    correction_bias?,
    token_ids?,
    token_table?,
    token_table_offset~,
  )
  let (routed, shared, finalize, _) = self.bind_experts(
    context,
    aot_module,
    counts~,
    input~,
    selected_indices=indices,
    selected_weights=weights,
    routed_bank~,
    shared_bank~,
    output~,
    prefix~,
  )
  self.launches.append(routed)
  self.launches.append(shared)
  self.launches.push(finalize)
}

///|
pub fn MoeProgram::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard !self.closed && !self.launches.is_empty() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}
