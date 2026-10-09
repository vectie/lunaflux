///|
/// Prepare a reusable BF16 AdaLN conditioning table from F32 timestep MLP
/// output. Append once before the first block in an evaluation, not per layer.
/// Caller owns disjoint input/output regions and the exact-shape AOT function.
pub fn conditioning_activation_launch(
  function : @device.Function,
  elements~ : Int64,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let dimensions = linear_dimensions(elements, 256)
  @device.OrderedKernelLaunch::new(
    function,
    dimensions,
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        input,
        offset=input_offset,
        byte_count=elements * 4L,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        output,
        offset=output_offset,
        byte_count=elements * 2L,
        alignment=2L,
      ),
    ]),
  )
}
