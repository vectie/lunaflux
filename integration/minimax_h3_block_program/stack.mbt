///|
fn validate_layer_order(
  ordinals : ArrayView[Int],
  expected : Int,
) -> Unit raise @device.DeviceError {
  guard expected > 0 && ordinals.length() == expected else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  for index, ordinal in ordinals {
    guard index == ordinal else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
}

///|
/// Assemble all transformer layers followed by both output heads. A single
/// block workspace and parameter arena are reused in strict layer order.
/// This list neither initializes packed state nor computes timesteps; those
/// producers precede it. Both flow updates must follow the complete list.
/// Weight arenas are borrowed and must remain resident until frame release.
pub fn Program::bind_stack(
  self : Self,
  output : OutputProgram,
  storage : @joint_diffusion_device.LatentStorage,
  rotary : @joint_diffusion_device.RotaryBindings,
  block_modulation : @joint_diffusion_device.ModulationBindings,
  final_modulation : @joint_diffusion_device.ModulationBindings,
  layers : ArrayView[
    (@minimax_h3_weights.BlockWeightPacking, @device.Allocation),
  ],
  output_packing : @minimax_h3_weights.OutputWeightPacking,
  output_weights~ : @device.Allocation,
  conditioning~ : @device.Allocation,
  conditioning_offset~ : Int64,
  state~ : @joint_diffusion_device.PackedHidden,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise {
  validate_layer_order(
    layers.map(pair => pair.0.layer),
    self.source.shape.layer_count(),
  )
  guard self.source.model.plan_digest() == output.source.model.plan_digest() &&
    self.source.rows == output.source.packed_rows &&
    self.source.packed_mapping == output.source.packed_mapping &&
    self.source.distinct_timesteps == output.source.distinct_timesteps else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  state.with_state(
    self.source.packed_mapping,
    width=self.source.shape.hidden_size(),
    (allocation, offset) => {
      let launches : Array[@device.OrderedKernelLaunch] = []
      for (packing, weights) in layers {
        launches.append(
          self.bind(
            rotary,
            block_modulation,
            packing,
            conditioning~,
            conditioning_offset~,
            state=allocation,
            state_offset=offset,
            weights~,
            parameters~,
            parameter_offset~,
          ),
        )
      }
      launches.append(
        output.bind(
          storage,
          final_modulation,
          output_packing,
          conditioning~,
          conditioning_offset~,
          state=allocation,
          state_offset=offset,
          weights=output_weights,
        ),
      )
      launches
    },
  )
}
