///|
fn selected_extent(
  start : Int,
  count : Int,
  rows : Int,
  width : Int,
) -> Int64 raise @device.DeviceError {
  guard start >= 0 &&
    count > 0 &&
    start <= rows &&
    count <= rows - start &&
    width > 0 &&
    count.to_int64() <= 2147483647L / width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  count.to_int64() * width.to_int64()
}

///|
/// Final modulation selects a contiguous media segment before F32 projection.
/// This map indexes distinct timesteps, NOT block timestep/modality pairs.
/// The kernel is compiled for selected_rows, width and parameter_rows.
pub fn ModulationBindings::selected_output_launch(
  self : Self,
  function : @device.Function,
  start_row~ : Int,
  selected_rows~ : Int,
  normalized~ : @device.Allocation,
  normalized_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let elements = selected_extent(
    start_row,
    selected_rows,
    self.rows,
    self.width,
  )
  let skip = start_row.to_int64() * self.width.to_int64() * 2L
  let table_bytes = self.parameter_rows.to_int64() * self.width.to_int64() * 2L
  guard normalized_offset >= 0L &&
    normalized_offset <= 9223372036854775807L - skip &&
    parameter_offset >= 0L &&
    parameter_offset <= 9223372036854775807L - table_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let arguments = @device.KernelArguments::new([
    @device.KernelArgument::device_region(
      normalized,
      offset=normalized_offset + skip,
      byte_count=elements * 2L,
      alignment=2L,
    ),
    @device.KernelArgument::device_region(
      parameters,
      offset=parameter_offset,
      byte_count=table_bytes,
      alignment=2L,
    ),
    @device.KernelArgument::device_region(
      parameters,
      offset=parameter_offset + table_bytes,
      byte_count=table_bytes,
      alignment=2L,
    ),
    @device.KernelArgument::device_region(
      self.indices,
      offset=self.offset + start_row.to_int64() * 4L,
      byte_count=selected_rows.to_int64() * 4L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      output,
      offset=output_offset,
      byte_count=elements * 4L,
      alignment=4L,
    ),
  ])
  @device.OrderedKernelLaunch::new(
    function,
    linear_dimensions(elements, 256),
    arguments,
  )
}
