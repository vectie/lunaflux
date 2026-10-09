///|
fn reference_projection_launch(
  function : @device.Function,
  copy : @minimax_h3_block_source.ConditioningCopy,
  hidden~ : Int,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  weights~ : @device.Allocation,
  packing~ : @minimax_h3_weights.InputWeightPacking,
  output~ : @device.Allocation,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let (weight_offset, bias_offset) = match copy.latent {
    Some(VideoLatent) => (packing.video_offset, packing.video_bias_offset)
    Some(AudioLatent) => (packing.audio_offset, packing.audio_bias_offset)
    None => raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let elements = copy.byte_count / 2L
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=((elements + 255L) / 256L).to_int(),
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        input,
        offset=input_offset,
        byte_count=copy.rows.to_int64() * copy.input_width.to_int64() * 4L,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        weights,
        offset=weight_offset,
        byte_count=hidden.to_int64() * copy.input_width.to_int64() * 4L,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        weights,
        offset=bias_offset,
        byte_count=hidden.to_int64() * 4L,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        output,
        offset=copy.destination_offset,
        byte_count=copy.byte_count,
        alignment=2L,
      ),
    ]),
  )
}
