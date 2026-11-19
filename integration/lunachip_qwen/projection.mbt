///|
pub(all) suberror ProjectionError {
  InvalidProjection
  InvalidWorkspace
  Lowering(@compiler.LoweringError)
}

///|
fn workspace_range(
  workspace : @layout.Region,
  address : UInt64,
  bytes : UInt64,
) -> Unit raise ProjectionError {
  if address < workspace.address ||
    address - workspace.address > workspace.bytes ||
    bytes > workspace.bytes - (address - workspace.address) {
    raise InvalidWorkspace
  }
}

///|
/// Prepare a contiguous row tile of an existing BF16 Qwen matrix. The tied
/// embedding/output uses ordinal zero. Source weights stay in their original
/// row-major DDR allocation; no full-matrix conversion or copy is introduced.
/// Input and output are caller-owned FP32 workspace ranges. This does not load
/// weights, schedule layers, normalize activations, or launch a device.
pub fn prepare_projection(
  device : @device.DeviceContract,
  layout : @layout.Layout,
  ordinal : Int32,
  first_row : Int32,
  rows : Int32,
  activation_address : UInt64,
  output_address : UInt64,
) -> @device.PreparedProgram raise ProjectionError {
  guard layout.weight(ordinal) is Some(weight) &&
    layout.dimensions(ordinal) is Some(shape) else {
    raise InvalidProjection
  }
  if shape.length() != 2 ||
    first_row < 0 ||
    rows < 1 ||
    rows > 4096 ||
    first_row.to_int64() > shape[0] ||
    rows.to_int64() > shape[0] - first_row.to_int64() ||
    shape[1] < 64L ||
    shape[1] > 4096L ||
    shape[1] % 64L != 0L {
    raise InvalidProjection
  }
  workspace_range(
    layout.workspace(),
    activation_address,
    shape[1].reinterpret_as_uint64() * 4UL,
  )
  workspace_range(layout.workspace(), output_address, rows.to_uint64() * 4UL)
  let weights_id = @tile.LunaTileTensorId::from_stable_int(1) catch {
    _ => raise InvalidProjection
  }
  let activation_id = @tile.LunaTileTensorId::from_stable_int(2) catch {
    _ => raise InvalidProjection
  }
  let output_id = @tile.LunaTileTensorId::from_stable_int(3) catch {
    _ => raise InvalidProjection
  }
  let operation = @compiler.OrderedBf16Gemv::new(
    weights=@tile.LunaTileTensorView::new(
      id=weights_id,
      dtype=BFloat16,
      address_space=Global,
      shape=[rows.to_int64(), shape[1]],
      strides_bytes=[shape[1] * 2L, 2L],
      offset_bytes=first_row.to_int64() * shape[1] * 2L,
      alignment_bytes=128L,
    ),
    activation=@tile.LunaTileTensorView::new(
      id=activation_id,
      dtype=Float32,
      address_space=Global,
      shape=[shape[1]],
      strides_bytes=[4L],
      offset_bytes=0L,
      alignment_bytes=4L,
    ),
    output=@tile.LunaTileTensorView::new(
      id=output_id,
      dtype=Float32,
      address_space=Global,
      shape=[rows.to_int64()],
      strides_bytes=[4L],
      offset_bytes=0L,
      alignment_bytes=4L,
    ),
  ) catch {
    e => raise Lowering(e)
  }
  operation.lower(device, [
    { tensor: weights_id, address: weight.address, },
    { tensor: activation_id, address: activation_address, },
    { tensor: output_id, address: output_address, },
  ]) catch {
    e => raise Lowering(e)
  }
}
