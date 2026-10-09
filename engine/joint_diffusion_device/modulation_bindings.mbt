///|
pub fn ModulationBindings::parameter_row_count(self : Self) -> Int {
  self.parameter_rows
}

///|
pub fn ModulationBindings::matches_geometry(
  self : Self,
  rows~ : Int,
  width~ : Int,
  parameter_rows~ : Int,
) -> Bool {
  self.rows == rows &&
  self.width == width &&
  self.parameter_rows == parameter_rows
}

///|
/// Borrowed immutable row indices for compact modulation tables. Prepare once
/// per distinct timestep equality pattern, not once per layer or execution.
pub struct ModulationBindings {
  priv indices : @device.Allocation
  priv offset : Int64
  priv rows : Int
  priv width : Int
  priv parameter_rows : Int
}

///|
fn modulation_index_bytes(
  indices : ArrayView[Int],
  parameter_rows : Int,
) -> Bytes raise @device.DeviceError {
  guard indices.length() > 0 &&
    indices.length() <= 536870911 &&
    parameter_rows > 0 else {
    raise @device.DeviceError::InvalidArgument(CopyToDevice)
  }
  for index in indices {
    guard index >= 0 && index < parameter_rows else {
      raise @device.DeviceError::InvalidArgument(CopyToDevice)
    }
  }
  let bytes : Array[Byte] = Array::make(indices.length() * 4, 0)
  for i, index in indices {
    let bits = index.reinterpret_as_uint()
    for b in 0..<4 {
      bytes[i * 4 + b] = ((bits >> (b * 8)) & 255U).to_byte()
    }
  }
  Bytes::from_array(bytes)
}

///|
/// Allocation remains caller-owned on success or failure. No queued frame may
/// use this region until upload completes; retry failed uploads in full.
pub fn ModulationBindings::upload(
  context : @device.Context,
  allocation : @device.Allocation,
  indices : ArrayView[Int],
  offset~ : Int64,
  width~ : Int,
  parameter_rows~ : Int,
) -> Self raise @device.DeviceError {
  guard width > 0 &&
    parameter_rows > 0 &&
    indices.length().to_int64() <= 2147483647L / width.to_int64() &&
    parameter_rows.to_int64() <= 2147483647L / width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(CopyToDevice)
  }
  let bytes = modulation_index_bytes(indices, parameter_rows)
  context.validate_allocation_region(
    allocation,
    offset~,
    byte_count=bytes.length().to_int64(),
    alignment=4L,
  )
  allocation.copy_from_host(bytes, 0L, offset, bytes.length().to_int64())
  {
    indices: allocation,
    offset,
    rows: indices.length(),
    width,
    parameter_rows,
  }
}

///|
/// Bind affine(x,shift,scale,indices,out) or residual(x,gate,branch,indices,out).
/// For affine, third is a compact parameter table; for residual it is a full
/// tensor. The caller supplies the matching admitted function. Storage is
/// borrowed and must not overlap immutable indices/parameters with outputs.
pub fn ModulationBindings::launch(
  self : Self,
  function : @device.Function,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  parameter~ : @device.Allocation,
  parameter_offset~ : Int64,
  third~ : @device.Allocation,
  third_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  residual~ : Bool,
) -> @device.OrderedKernelLaunch raise @device.DeviceError {
  let tensor_bytes = self.rows.to_int64() * self.width.to_int64() * 2L
  let table_bytes = self.parameter_rows.to_int64() * self.width.to_int64() * 2L
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
    region(parameter, parameter_offset, table_bytes),
    region(
      third,
      third_offset,
      if residual {
        tensor_bytes
      } else {
        table_bytes
      },
    ),
    @device.KernelArgument::device_region(
      self.indices,
      offset=self.offset,
      byte_count=self.rows.to_int64() * 4L,
      alignment=4L,
    ),
    region(output, output_offset, tensor_bytes),
  ])
  @device.OrderedKernelLaunch::new(
    function,
    linear_dimensions(tensor_bytes / 2L, 256),
    arguments,
  )
}
