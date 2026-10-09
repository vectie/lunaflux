///|
/// Startup-only geometry for staged packed SwiGLU. Product storage is reusable
/// across serialized layers; it must remain distinct from input and output.
pub struct StagedMlpLayout {
  input_bytes : Int64
  product_bytes : Int64
  packed_weight_bytes : Int64
  down_weight_bytes : Int64
}

///|
pub fn StagedMlpLayout::new(
  rows~ : Int,
  hidden_width~ : Int,
  intermediate_width~ : Int,
) -> Self raise @device.DeviceError {
  guard rows > 0 &&
    hidden_width > 0 &&
    intermediate_width > 0 &&
    intermediate_width <= 1073741823 &&
    rows.to_int64() <= 2147483647L / hidden_width.to_int64() &&
    rows.to_int64() <= 2147483647L / intermediate_width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let matrix = hidden_width.to_int64() * intermediate_width.to_int64()
  guard matrix <= 9223372036854775807L / 6L else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  {
    input_bytes: rows.to_int64() * hidden_width.to_int64() * 2L,
    product_bytes: rows.to_int64() * intermediate_width.to_int64() * 2L,
    packed_weight_bytes: matrix * 4L,
    down_weight_bytes: matrix * 2L,
  }
}

///|
/// Bind prefix_gate_up then prefix_down from staged_packed_swiglu_bf16_source.
/// Weights contain contiguous [up;gate] followed by down. All allocations are
/// borrowed; the caller must prevent scratch aliasing live input/weights/output
/// and release queues before storage. No allocation, upload or submission here.
pub fn StagedMlpLayout::launches(
  self : Self,
  gate_up : @device.Function,
  down : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  scratch~ : @device.Allocation,
  scratch_offset~ : Int64,
  weights~ : @device.Allocation,
  weight_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
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
  ignore(
    region(
      weights,
      weight_offset,
      self.packed_weight_bytes + self.down_weight_bytes,
    ),
  )
  let product = region(scratch, scratch_offset, self.product_bytes)
  let first = @device.OrderedKernelLaunch::new(
    gate_up,
    linear_dimensions(self.product_bytes / 2L, 256),
    @device.KernelArguments::new([
      region(input, input_offset, self.input_bytes),
      region(weights, weight_offset, self.packed_weight_bytes),
      product,
    ]),
  )
  let second = @device.OrderedKernelLaunch::new(
    down,
    linear_dimensions(self.input_bytes / 2L, 256),
    @device.KernelArguments::new([
      product,
      region(
        weights,
        weight_offset + self.packed_weight_bytes,
        self.down_weight_bytes,
      ),
      region(output, output_offset, self.input_bytes),
    ]),
  )
  [first, second]
}
