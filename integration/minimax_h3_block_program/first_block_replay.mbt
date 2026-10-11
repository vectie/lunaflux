///|
/// Offline first-block replay with the exact prepared first-step mapping.
/// Execute existing timestep/input producers first and retire before reading.
/// Ordinary serving keeps bind_predictions, with no observers or readback.
pub fn PredictionMetadata::bind_first_block(
  self : Self,
  variants : ArrayView[PredictionVariant],
  input : InputProgram,
  rotary : @joint_diffusion_device.RotaryBindings,
  weights : PredictionWeights,
  timestep_embeddings~ : @device.Allocation,
  parameters~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise {
  guard self.ready else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let step = self.schedule.steps()[0]
  let variant = for variant in variants {
    if variant.timestep.source.rows == step.values.length() {
      break variant
    }
  } nobreak {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let (packing, allocation) = weights.layers[0]
  input
  .packed_state()
  .with_state(
    self.sequence.assembly().mapping(),
    width=self.sequence.assembly().width(),
    (state, offset) => {
      variant.block.bind(
        rotary,
        self.blocks[step.pattern],
        packing,
        conditioning=timestep_embeddings,
        conditioning_offset=0L,
        state~,
        state_offset=offset,
        weights=allocation,
        parameters~,
        parameter_offset=0L,
      )
    },
  )
}
