///|
pub struct Head {
  source : Bytes
  symbols : ReadOnlyArray[String]
  rows : Int
  hidden_bytes : Int64
  output_bytes : Int64
  weights : @minimax_h3_weights.VideoFrontendWeightPacking
}

///|
pub fn Head::new(
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  prefix~ : String,
) -> Self raise {
  let source = @video_latent_embedding_cuda_source.decoder_head_source(
    temporal~,
    height~,
    width~,
    hidden=2048,
    patch_t=4,
    patch_hw=16,
    channels=3,
    prefix~,
  )
  let positions = temporal * height * width
  {
    source,
    symbols: ReadOnlyArray::from_array([
      prefix + "_head_norm",
      prefix + "_head_unpack",
    ]),
    rows: positions + 5,
    hidden_bytes: (positions.to_int64() + 5L) * 2048L * 4L,
    output_bytes: positions.to_int64() * 3072L * 4L,
    weights: @minimax_h3_weights.video_head_weight_packing(max_bytes=25194496L),
  }
}

///|
pub fn Head::launches(
  self : Self,
  functions : ArrayView[@device.Function],
  input : @device.Allocation,
  weights : @device.Allocation,
  scratch : @device.Allocation,
  output : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 2 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn region(
    a : @device.Allocation,
    o : Int64,
    n : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      a,
      offset=o,
      byte_count=n,
      alignment=4L,
    )
  }
  fn weight(i : Int) -> @device.KernelArgument raise @device.DeviceError {
    let copy = self.weights.copies[i]
    region(weights, copy.destination_offset, copy.byte_count)
  }
  [
    @device.OrderedKernelLaunch::new(
      functions[0],
      @device.KernelLaunchDimensions::new(
        grid_x=self.rows,
        grid_y=1,
        grid_z=1,
        block_x=1,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      @device.KernelArguments::new([
        region(input, 0L, self.hidden_bytes),
        weight(0),
        weight(1),
        region(scratch, 0L, self.hidden_bytes),
      ]),
    ),
    @device.OrderedKernelLaunch::new(
      functions[1],
      dimensions(self.output_bytes),
      @device.KernelArguments::new([
        region(scratch, 0L, self.hidden_bytes),
        weight(2),
        weight(3),
        region(output, 0L, self.output_bytes),
      ]),
    ),
  ]
}
