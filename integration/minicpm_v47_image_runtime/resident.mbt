///|
/// Combined model residency. Uses the existing bounded text/vision loaders;
/// one scratch buffer is reused sequentially during startup.
pub struct Resident {
  priv budget : Budget
  priv text : @weights.Loaded
  priv vision : @vision.Loaded
  priv execution : Execution
  priv mut state : SessionState
}

///|
pub fn Resident::new(
  program : @multimodal.Program,
  vision : @vision.Program,
  text_weights : @weights.Plan,
  vision_weights : @vision.WeightPlan,
) -> Self raise {
  {
    budget: Budget::new(
      program,
      vision,
      text_weight_bytes=text_weights.byte_count(),
      vision_weight_bytes=vision_weights.byte_count(),
    ),
    text: @weights.Loaded::new(text_weights),
    vision: @vision.Loaded::new(vision_weights),
    execution: Execution::new(program, vision),
    state: Fresh,
  }
}

///|
pub fn Resident::budget(self : Self) -> Budget {
  self.budget
}

///|
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
  self.text.prepare(
    context,
    root,
    maximum_device_bytes=self.budget.text.weight_bytes,
    scratch~,
    limits~,
  )
  self.vision.prepare(
    context,
    root,
    maximum_device_bytes=self.budget.vision_weight_bytes,
    scratch~,
    limits~,
  )
  self.execution.prepare(
    context,
    images,
    layers=self.text.layer_weights(),
    text_weights=self.text.text_weights(),
    vision_weights=self.vision.weights(),
    maximum_mutable_bytes=self.budget.mutable_bytes,
    policy~,
  )
  self.state = Prepared
}

///|
/// Borrowed driver. Do not independently prepare/close it; close Resident.
pub fn Resident::execution(self : Self) -> Execution raise {
  guard self.state is Prepared else { raise ExecutionError::InvalidLifecycle }
  self.execution
}

///|
pub fn Resident::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.execution.close()
  self.vision.close()
  self.text.close()
  self.state = Closed
}
