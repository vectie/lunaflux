///|
/// Bind the staged QK-normalization/rotary ABI during frame preparation.
/// Inputs and outputs each contain two contiguous BF16 tensors (Q then K).
/// Weights contain the two contiguous per-head normalization vectors.
/// All allocations are borrowed. The function must be compiled for these exact
/// dimensions and the numerical contract of joint_attention_cuda_source.
pub fn RotaryBindings::qk_norm_launch(
  self : Self,
  function : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  head_count~ : Int,
  head_dimension~ : Int,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let (tensor_bytes, vector_bytes, dimensions) = qk_rotary_geometry(
    self.rows,
    self.width,
    head_count,
    head_dimension,
  )
  fn region(
    allocation : @device.Allocation,
    offset : Int64,
    size : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      allocation,
      offset~,
      byte_count=size,
      alignment=2L,
    )
  }
  let arguments = @device.KernelArguments::new([
    region(input, input_offset, tensor_bytes),
    region(input, input_offset + tensor_bytes, tensor_bytes),
    region(weights, weight_offset, vector_bytes),
    region(weights, weight_offset + vector_bytes, vector_bytes),
    self.cosine(),
    self.sine(),
    region(output, output_offset, tensor_bytes),
    region(output, output_offset + tensor_bytes, tensor_bytes),
  ])
  @device.OrderedKernelLaunch::new(function, dimensions, arguments)
}

///|
fn qk_rotary_geometry(
  rows : Int,
  rotary_width : Int,
  heads : Int,
  dimension : Int,
) -> (Int64, Int64, @device.KernelLaunchDimensions) raise @device.DeviceError {
  guard rows > 0 &&
    heads > 0 &&
    dimension > 0 &&
    rotary_width > 0 &&
    rotary_width % 2 == 0 &&
    rotary_width <= dimension else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let row_heads = rows.to_int64() * heads.to_int64()
  guard row_heads <= 2147483647L / dimension.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  // The current staged renderer assigns one head to one single-thread CTA.
  // This is correctness-path geometry, not a performance-qualified schedule.
  (
    row_heads * dimension.to_int64() * 2L,
    dimension.to_int64() * 2L,
    linear_dimensions(row_heads, 1),
  )
}
