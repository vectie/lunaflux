///|
/// Prepare text conditioning once per request: biased BF16 context projection
/// followed by the refiner. Encoder input and projected state must be separate
/// allocations, since projection cannot safely overwrite its input matrix.
/// The caller owns buffers and supplies an admitted context weight allocation
/// packed at offset zero (upload_context destination_offset=0).
pub fn RefinerProgram::bind_context(
  self : Self,
  layers : ArrayView[
    (@minimax_h3_weights.RefinerWeightPacking, @device.Allocation),
  ],
  encoded~ : @device.Allocation,
  encoded_offset~ : Int64,
  weights~ : @device.Allocation,
  state~ : @device.Allocation,
  state_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let packing = @minimax_h3_weights.context_weight_packing(
    self.source.model,
    max_bytes=weights.byte_count(),
  ) catch {
    _ => raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let refiner = self.bind(layers, state~, state_offset~)
  let shape = self.source.model.denoiser()
  let layout = @joint_diffusion_device.ParameterProjectionLayout::new(
    rows=self.source.rows,
    input_width=shape.text_conditioning_dimension(),
    width=shape.hidden_size(),
    groups=1,
    fields=1,
  )
  let launches = [
    layout.launch(
      self.functions[9],
      input=encoded,
      input_offset=encoded_offset,
      weights~,
      weight_offset=packing.weight_offset,
      bias_offset=packing.bias_offset,
      output=state,
      output_offset=state_offset,
    ),
  ]
  launches.append(refiner)
  launches
}
