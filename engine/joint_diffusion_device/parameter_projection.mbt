///|
pub struct ParameterProjectionLayout {
  input_bytes : Int64
  weight_bytes : Int64
  bias_bytes : Int64
  output_bytes : Int64
}

///|
pub fn ParameterProjectionLayout::new(
  rows~ : Int,
  input_width~ : Int,
  width~ : Int,
  groups~ : Int,
  fields~ : Int,
) -> Self raise @device.DeviceError {
  guard rows > 0 && input_width > 0 && width > 0 && groups > 0 && fields > 0 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let group_fields = groups.to_int64() * fields.to_int64()
  guard group_fields <= 2147483647L / width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let outputs = group_fields * width.to_int64()
  guard rows.to_int64() <= 2147483647L / outputs else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  {
    input_bytes: rows.to_int64() * input_width.to_int64() * 2L,
    weight_bytes: outputs * input_width.to_int64() * 2L,
    bias_bytes: outputs * 2L,
    output_bytes: rows.to_int64() * outputs * 2L,
  }
}

///|
/// Bind grouped_parameter_projection_source. Input is BF16; this projection
/// applies no activation. AdaLN callers supply already SiLU-activated input.
/// A single group/field also expresses plain biased row-major projection.
pub fn ParameterProjectionLayout::launch(
  self : Self,
  function : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
  bias_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
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
    linear_dimensions(self.output_bytes / 2L, 256),
    @device.KernelArguments::new([
      region(input, input_offset, self.input_bytes),
      region(weights, weight_offset, self.weight_bytes),
      region(weights, bias_offset, self.bias_bytes),
      region(output, output_offset, self.output_bytes),
    ]),
  )
}
