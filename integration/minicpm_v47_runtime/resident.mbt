///|
/// Whole text-model residency. The external context remains borrowed; this
/// owner combines streaming weights with the independent execution session.
pub struct Resident {
  priv budget : Budget
  priv weights : @weights.Loaded
  priv execution : Execution
  priv mut state : SessionState
}

///|
pub fn Resident::new(
  program : @text.Program,
  weights : @weights.Plan,
) -> Self raise {
  {
    budget: Budget::for_text(program, weight_bytes=weights.byte_count()),
    weights: @weights.Loaded::new(weights),
    execution: Execution::new(program),
    state: Fresh,
  }
}

///|
pub fn Resident::budget(self : Self) -> Budget {
  self.budget
}

///|
/// One startup-only bounded transfer. Admission covers weights plus all known
/// mutable/output allocations before any allocation or module load occurs.
pub fn Resident::prepare(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  images : ModuleImages,
  maximum_device_bytes~ : Int64,
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
  policy~ : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Unit raise {
  guard self.state is Fresh &&
    maximum_device_bytes >= self.budget.total_allocation_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.state = Failed
  self.weights.prepare(
    context,
    root,
    maximum_device_bytes=self.budget.weight_bytes,
    scratch~,
    limits~,
  )
  self.execution.prepare(
    context,
    images,
    layers=self.weights.layer_weights(),
    weights=self.weights.text_weights(),
    maximum_mutable_bytes=self.budget.storage_bytes + self.budget.output_bytes,
    policy~,
  )
  self.state = Prepared
}

///|
pub fn Resident::runner(self : Self) -> @request.Runner raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.execution.runner()
}

///|
/// Close queued execution before releasing the weights it borrows.
pub fn Resident::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.execution.close()
  self.weights.close()
  self.state = Closed
}
