///|
/// A complete routed/shared launch frame with an obligatory F32 reduction
/// before the returned publication index. A decoder must splice this boundary
/// into a collective-aware queue, not enqueue the frame as a local sequence.
pub struct ExpertTensorBoundary {
  priv before_kernel : Int
  priv partials : @device.Allocation
  priv width : Int
}

///|
pub fn ExpertTensorBoundary::before_kernel(self : Self) -> Int {
  self.before_kernel
}

///|
pub fn ExpertTensorBoundary::partials(self : Self) -> @device.Allocation {
  self.partials
}

///|
pub fn ExpertTensorBoundary::width(self : Self) -> Int {
  self.width
}

///|
pub fn MoeProgram::shared_tensor_boundary(
  self : Self,
) -> ExpertTensorBoundary? raise @device.DeviceError {
  guard !self.closed && !self.launches.is_empty() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.shared_tensor_boundary
}

///|
pub fn MoeProgram::prepare_routed_tensor_frame(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  maximum_workspace_bytes~ : Int64,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  router_weight~ : @device.Allocation,
  correction_bias? : @device.Allocation,
  routed_bank~ : @device.Allocation,
  shared_bank~ : @device.Allocation,
  output~ : @device.Allocation,
) -> ExpertTensorBoundary raise @device.DeviceError {
  self.check_fresh(maximum_workspace_bytes)
  let (rows, hidden, _, experts, selected) = self.routed.plan().geometry()
  guard self.routed.plan().down_publication() == ReduceFloat32BeforeRound &&
    self.routed.plan().local_experts().length() == experts &&
    hidden.to_int64() * selected.to_int64() <= 2147483647L &&
    rows > 0 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let (indices, weights, prefix) = self.bind_router(
    context,
    aot_module,
    counts~,
    input~,
    router_weight~,
    correction_bias?,
    token_table_offset=0L,
  )
  let (routed, shared, finalize, partials) = self.bind_experts(
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
  { before_kernel: routed.length() - 1, partials, width: hidden * selected, }
}
