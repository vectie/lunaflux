///|
pub struct TimestepMlpLayout {
  priv rows : Int
  priv input_width : Int
  priv hidden_width : Int
  priv output_width : Int
  scratch_bytes : Int64
  output_bytes : Int64
}

///|
pub fn TimestepMlpLayout::new(
  rows~ : Int,
  input_width~ : Int,
  hidden_width~ : Int,
  output_width~ : Int,
) -> Self raise @device.DeviceError {
  guard rows > 0 &&
    input_width > 0 &&
    hidden_width > 0 &&
    output_width > 0 &&
    rows.to_int64() <= 2147483647L / input_width.to_int64() &&
    rows.to_int64() <= 2147483647L / hidden_width.to_int64() &&
    rows.to_int64() <= 2147483647L / output_width.to_int64() &&
    hidden_width.to_int64() <=
    9223372036854775807L / 4L / input_width.to_int64() &&
    hidden_width.to_int64() <=
    9223372036854775807L / 4L / output_width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  {
    rows,
    input_width,
    hidden_width,
    output_width,
    scratch_bytes: rows.to_int64() * hidden_width.to_int64() * 4L,
    output_bytes: rows.to_int64() * output_width.to_int64() * 4L,
  }
}

///|
/// Bind staged dense+SiLU then dense. All buffers remain caller-owned and
/// scratch must not alias input/output/weights. The functions are AOT compiled
/// for this geometry; binding does not allocate, submit or synchronize.
pub fn TimestepMlpLayout::launches(
  self : Self,
  hidden_function : @device.Function,
  output_function : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  scratch~ : @device.Allocation,
  scratch_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  weights~ : @device.Allocation,
  first_weight_offset~ : Int64,
  first_bias_offset~ : Int64,
  second_weight_offset~ : Int64,
  second_bias_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  fn region(
    a : @device.Allocation,
    offset : Int64,
    bytes : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      a,
      offset~,
      byte_count=bytes,
      alignment=4L,
    )
  }
  let temporary = region(scratch, scratch_offset, self.scratch_bytes)
  let first = @device.OrderedKernelLaunch::new(
    hidden_function,
    linear_dimensions(self.scratch_bytes / 4L, 256),
    @device.KernelArguments::new([
      region(
        input,
        input_offset,
        self.rows.to_int64() * self.input_width.to_int64() * 4L,
      ),
      region(
        weights,
        first_weight_offset,
        self.input_width.to_int64() * self.hidden_width.to_int64() * 4L,
      ),
      region(weights, first_bias_offset, self.hidden_width.to_int64() * 4L),
      temporary,
    ]),
  )
  let second = @device.OrderedKernelLaunch::new(
    output_function,
    linear_dimensions(self.output_bytes / 4L, 256),
    @device.KernelArguments::new([
      temporary,
      region(
        weights,
        second_weight_offset,
        self.hidden_width.to_int64() * self.output_width.to_int64() * 4L,
      ),
      region(weights, second_bias_offset, self.output_width.to_int64() * 4L),
      region(output, output_offset, self.output_bytes),
    ]),
  )
  [first, second]
}
