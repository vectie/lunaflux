///|
/// Bind the fixed-row BF16 RMSNorm renderer (one 256-thread CTA per row).
/// Epsilon and reduction semantics belong to the admitted function, not this
/// runtime adapter. Buffers are borrowed and weights must not overlap output.
pub fn block_rms_norm_launch(
  function : @device.Function,
  rows~ : Int,
  width~ : Int,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  guard rows > 0 &&
    width > 0 &&
    rows.to_int64() <= 2147483647L / width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let bytes = rows.to_int64() * width.to_int64() * 2L
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=rows,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  fn region(
    allocation : @device.Allocation,
    offset : Int64,
    bytes : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      allocation,
      offset~,
      byte_count=bytes,
      alignment=2L,
    )
  }
  @device.OrderedKernelLaunch::new(
    function,
    dimensions,
    @device.KernelArguments::new([
      region(input, input_offset, bytes),
      region(weights, weight_offset, width.to_int64() * 2L),
      region(output, output_offset, bytes),
    ]),
  )
}

///|
fn[T] compose_block(
  attention_prefix : ArrayView[T],
  attention : ArrayView[T],
  attention_residual : T,
  mlp_prefix : ArrayView[T],
  mlp : ArrayView[T],
  mlp_residual : T,
) -> Array[T] raise @device.DeviceError {
  guard attention_prefix.length() == 2 &&
    attention.length() == 4 &&
    mlp_prefix.length() == 2 &&
    mlp.length() == 2 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let result = attention_prefix.to_array()
  result.append(attention)
  result.push(attention_residual)
  result.append(mlp_prefix)
  result.append(mlp)
  result.push(mlp_residual)
  result
}

///|
/// Compose the twelve ordered launches of a staged joint transformer block.
/// Prefixes are [RMSNorm, indexed affine]; attention and MLP come from their
/// matching staged adapters. Callers bind matching shapes, weights and storage
/// regions first. Append this list to the denoising frame before RF updates;
/// do not create a separate synchronized queue per block.
pub fn transformer_block_launches(
  attention_prefix : ArrayView[@device.OrderedKernelLaunch],
  attention : ArrayView[@device.OrderedKernelLaunch],
  attention_residual : @device.OrderedKernelLaunch,
  mlp_prefix : ArrayView[@device.OrderedKernelLaunch],
  mlp : ArrayView[@device.OrderedKernelLaunch],
  mlp_residual : @device.OrderedKernelLaunch,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  compose_block(
    attention_prefix, attention, attention_residual, mlp_prefix, mlp, mlp_residual,
  )
}
