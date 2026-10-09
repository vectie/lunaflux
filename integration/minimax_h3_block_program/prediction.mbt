///|
/// Immutable weights borrowed by the full prediction binder. Their packing
/// comes from the admitted model; no weight upload occurs per denoising step.
pub struct PredictionWeights {
  timestep : (@minimax_h3_weights.TimestepWeightPacking, @device.Allocation)
  input : (@minimax_h3_weights.InputWeightPacking, @device.Allocation)
  layers : ReadOnlyArray[
    (@minimax_h3_weights.BlockWeightPacking, @device.Allocation),
  ]
  output : (@minimax_h3_weights.OutputWeightPacking, @device.Allocation)
}

///|
pub fn PredictionWeights::new(
  timestep : (@minimax_h3_weights.TimestepWeightPacking, @device.Allocation),
  input : (@minimax_h3_weights.InputWeightPacking, @device.Allocation),
  layers : ArrayView[
    (@minimax_h3_weights.BlockWeightPacking, @device.Allocation),
  ],
  output : (@minimax_h3_weights.OutputWeightPacking, @device.Allocation),
) -> Self {
  {
    timestep,
    input,
    layers: ReadOnlyArray::from_array(layers.to_owned()),
    output,
  }
}

///|
/// Complete prediction, ordered timestep -> latent input -> 50 blocks -> both
/// heads. Flow updates are deliberately appended only by PipelineProgram after
/// both predictions exist. Context rows and timestep embeddings are distinct.
pub fn Program::bind_prediction(
  self : Self,
  timestep : TimestepProgram,
  input : InputProgram,
  output : OutputProgram,
  storage : @joint_diffusion_device.LatentStorage,
  rotary : @joint_diffusion_device.RotaryBindings,
  block_modulation : @joint_diffusion_device.ModulationBindings,
  final_modulation : @joint_diffusion_device.ModulationBindings,
  weights : PredictionWeights,
  timestep_values~ : @device.Allocation,
  timestep_values_offset~ : Int64,
  context_rows~ : @device.Allocation,
  context_offset~ : Int64,
  timestep_embeddings~ : @device.Allocation,
  timestep_embeddings_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
) -> ReadOnlyArray[@device.OrderedKernelLaunch] raise {
  guard input.matches_storage(storage) &&
    self.source.model.plan_digest() == input.source.model.plan_digest() &&
    timestep.source.rows == self.source.distinct_timesteps &&
    timestep.source.input_width ==
    self.source.shape.timestep_frequency_dimension() &&
    timestep.source.hidden_width ==
    self.source.shape.timestep_embedding_hidden_dimension() &&
    timestep.source.output_width ==
    self.source.shape.final_adaln_conditioning_dimension() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let state = input.packed_state()
  let launches = timestep.bind(
    weights.timestep.0,
    timesteps=timestep_values,
    timestep_offset=timestep_values_offset,
    weights=weights.timestep.1,
    conditioning=timestep_embeddings,
    conditioning_offset=timestep_embeddings_offset,
  )
  launches.append(
    input.bind(
      weights.input.0,
      weights=weights.input.1,
      conditioning=Some(context_rows),
      conditioning_offset=context_offset,
    ),
  )
  launches.append(
    self.bind_stack(
      output,
      storage,
      rotary,
      block_modulation,
      final_modulation,
      weights.layers[:],
      weights.output.0,
      output_weights=weights.output.1,
      conditioning=timestep_embeddings,
      conditioning_offset=timestep_embeddings_offset,
      state~,
      parameters~,
      parameter_offset~,
    ),
  )
  ReadOnlyArray::from_array(launches)
}
