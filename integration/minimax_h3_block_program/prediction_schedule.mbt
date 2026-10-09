///|
/// An AOT geometry variant for a distinct-timestep count. These are borrowed
/// prepared programs, not JIT work. Scratch is shared only in serial frame order.
pub struct PredictionVariant {
  priv block : Program
  priv timestep : TimestepProgram
  priv output : OutputProgram
}

///|
pub fn PredictionVariant::new(
  block : Program,
  timestep : TimestepProgram,
  output : OutputProgram,
) -> Self {
  { block, timestep, output, }
}

///|
pub fn PredictionMetadata::matches_plan(
  self : Self,
  plan : @joint_diffusion_loop.PreparedPlan,
) -> Bool {
  self.plan.model_digest() == plan.model_digest() &&
  self.plan.request() == plan.request() &&
  self.plan.flow_table().bytes() == plan.flow_table().bytes()
}

///|
/// Typed conditioning join. It binds the future output of the prepared row
/// assembler but does not run it; the pipeline orders that producer first.
pub fn PredictionMetadata::bind_conditioned(
  self : Self,
  variants : ArrayView[PredictionVariant],
  input : InputProgram,
  storage : @joint_diffusion_device.LatentStorage,
  rotary : @joint_diffusion_device.RotaryBindings,
  weights : PredictionWeights,
  conditioning : ConditioningProgram,
  timestep_embeddings~ : @device.Allocation,
  timestep_embeddings_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
) -> ReadOnlyArray[ReadOnlyArray[@device.OrderedKernelLaunch]] raise {
  guard conditioning.source.model.plan_digest() == self.plan.model_digest() &&
    conditioning.source.sequence.segments() == self.sequence.segments() &&
    conditioning.source.sequence.assembly().mapping() ==
    self.sequence.assembly().mapping() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let (context_rows, context_offset) = conditioning.with_destination((a, offset) => {
    (a, offset)
  })
  self.bind_predictions(
    variants,
    input,
    storage,
    rotary,
    weights,
    context_rows~,
    context_offset~,
    timestep_embeddings~,
    timestep_embeddings_offset~,
    parameters~,
    parameter_offset~,
  )
}

///|
/// Bind every real plan step, choosing an already prepared AOT variant and the
/// exact immutable equality-pattern indices. No selection/upload occurs during
/// PipelineProgram::advance. Both RF updates are appended by that executor.
pub fn PredictionMetadata::bind_predictions(
  self : Self,
  variants : ArrayView[PredictionVariant],
  input : InputProgram,
  storage : @joint_diffusion_device.LatentStorage,
  rotary : @joint_diffusion_device.RotaryBindings,
  weights : PredictionWeights,
  context_rows~ : @device.Allocation,
  context_offset~ : Int64,
  timestep_embeddings~ : @device.Allocation,
  timestep_embeddings_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
) -> ReadOnlyArray[ReadOnlyArray[@device.OrderedKernelLaunch]] raise {
  guard self.ready &&
    self.allocation is Some(allocation) &&
    input.source.model.plan_digest() == self.plan.model_digest() &&
    input.source.sequence.assembly().width() == self.sequence.assembly().width() &&
    input.source.sequence.assembly().mapping() ==
    self.sequence.assembly().mapping() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let selected : Map[Int, PredictionVariant] = Map([])
  for variant in variants {
    let count = variant.timestep.source.rows
    guard !selected.contains(count) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    selected[count] = variant
  }
  let predictions : Array[ReadOnlyArray[@device.OrderedKernelLaunch]] = []
  for index, step in self.schedule.steps() {
    guard selected.get(step.values.length()) is Some(variant) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    predictions.push(
      variant.block.bind_prediction(
        variant.timestep,
        input,
        variant.output,
        storage,
        rotary,
        self.blocks[step.pattern],
        self.finals[step.pattern],
        weights,
        timestep_values=allocation,
        timestep_values_offset=self.layout.times[index],
        context_rows~,
        context_offset~,
        timestep_embeddings~,
        timestep_embeddings_offset~,
        parameters~,
        parameter_offset~,
      ),
    )
  }
  ReadOnlyArray::from_array(predictions)
}
