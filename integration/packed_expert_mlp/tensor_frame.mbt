///|
/// A complete routed/shared launch frame with an obligatory typed reduction
/// before the returned consumer index. A decoder must splice this boundary
/// into a collective-aware queue, not enqueue the frame as a local sequence.
pub struct ExpertTensorBoundary {
  priv before_kernel : Int32
  priv partials : @device.Allocation
  priv width : Int32
  priv kind : @device.TensorCollectiveKind
}

///|
pub fn ExpertTensorBoundary::before_kernel(self : Self) -> Int32 {
  self.before_kernel
}

///|
pub fn ExpertTensorBoundary::partials(self : Self) -> @device.Allocation {
  self.partials
}

///|
pub fn ExpertTensorBoundary::width(self : Self) -> Int32 {
  self.width
}

///|
pub fn ExpertTensorBoundary::kind(self : Self) -> @device.TensorCollectiveKind {
  self.kind
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
/// Independent producer coordinates are valid for joined tensor frames;
/// bind_experts owns distinct writable scratch for the two branches.
pub fn MoeProgram::producer_fork(self : Self) -> @parallel.ForkJoinSpan? {
  self.producer_fork
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
  transient_scratch? : @device.Allocation,
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
    transient_scratch?,
  )
  self.launches.append(routed)
  self.launches.append(shared)
  self.launches.push(finalize)
  {
    before_kernel: if self.reduction_placement == LocalCombinedBFloat16 {
      self.launches.length()
    } else {
      routed.length() - (if self.collective_region is Some(_) { 0 } else { 1 })
    },
    partials,
    width: self.combined_tensor_width().unwrap_or(hidden * selected),
    kind: self.tensor_kind(),
  }
}
