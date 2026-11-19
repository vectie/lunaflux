///|
/// Explicit mixed-precision operation, not a reinterpretation of LunaTile MMA.
/// y[r] = fold_k(add_rne, +0, mul_rne(exact_bf16(W[r,k]), x[k])).
/// All operands are contiguous global LunaTile views. Preparation is offline;
/// execution uses the retained LOAD/GEMV/STORE/HALT bytes without allocation.
pub struct OrderedBf16Gemv {
  priv weights : @tile.LunaTileTensorView
  priv activation : @tile.LunaTileTensorView
  priv output : @tile.LunaTileTensorView
  priv rows : Int32
  priv columns : Int32
}

///|
fn gemv_view(
  view : @tile.LunaTileTensorView,
  alignment : Int64,
) -> Unit raise LoweringError {
  let declared = view.alignment_bytes()
  if view.address_space() != Global ||
    view.offset_bytes() < 0L ||
    declared < alignment ||
    (declared & (declared - 1L)) != 0L ||
    view.offset_bytes() % declared != 0L {
    raise UnsupportedTensor
  }
}

///|
pub fn OrderedBf16Gemv::new(
  weights~ : @tile.LunaTileTensorView,
  activation~ : @tile.LunaTileTensorView,
  output~ : @tile.LunaTileTensorView,
) -> Self raise LoweringError {
  gemv_view(weights, 128L)
  gemv_view(activation, 4L)
  gemv_view(output, 4L)
  let shape = weights.shape()
  if weights.dtype() != BFloat16 ||
    shape.length() != 2 ||
    shape[0] < 1L ||
    shape[0] > 4096L ||
    shape[1] < 64L ||
    shape[1] > 4096L ||
    shape[1] % 64L != 0L {
    raise UnsupportedTensor
  }
  if weights.strides_bytes() != [shape[1] * 2L, 2L] ||
    activation.dtype() != Float32 ||
    output.dtype() != Float32 ||
    activation.shape() != [shape[1]] ||
    output.shape() != [shape[0]] ||
    activation.strides_bytes() != [4L] ||
    output.strides_bytes() != [4L] ||
    weights.id() == activation.id() ||
    weights.id() == output.id() ||
    activation.id() == output.id() {
    raise UnsupportedTensor
  }
  {
    weights,
    activation,
    output,
    rows: shape[0].to_int(),
    columns: shape[1].to_int(),
  }
}

///|
fn gemv_address(
  device : @lunachip.DeviceContract,
  tensor : @tile.LunaTileTensorView,
  bytes : UInt64,
  bindings : ArrayView[GlobalBinding],
) -> UInt64 raise LoweringError {
  let mut base : UInt64? = None
  for binding in bindings {
    if binding.tensor == tensor.id() {
      if base is Some(_) {
        raise InvalidBinding
      }
      base = Some(binding.address)
    }
  }
  guard base is Some(address) else { raise InvalidBinding }
  let offset = tensor.offset_bytes().reinterpret_as_uint64()
  device.check_range(address, offset) catch {
    e => raise Device(e)
  }
  let resolved = address + offset
  device.check_range(resolved, bytes) catch {
    e => raise Device(e)
  }
  if resolved % tensor.alignment_bytes().reinterpret_as_uint64() != 0UL {
    raise InvalidBinding
  }
  resolved
}

///|
/// Resolve allocation bases plus view offsets. All three DDR ranges must be
/// disjoint; output may not overwrite weights or activations used later.
pub fn OrderedBf16Gemv::lower(
  self : Self,
  device : @lunachip.DeviceContract,
  bindings : ArrayView[GlobalBinding],
) -> @lunachip.PreparedProgram raise LoweringError {
  if bindings.length() != 3 {
    raise InvalidBinding
  }
  let sizes = [
    self.rows.to_uint64() * self.columns.to_uint64() * 2UL,
    self.columns.to_uint64() * 4UL,
    self.rows.to_uint64() * 4UL,
  ]
  let views = [self.weights, self.activation, self.output]
  let addresses : Array[UInt64] = []
  for i, view in views {
    let address = gemv_address(device, view, sizes[i], bindings)
    for j, previous in addresses {
      if address < previous + sizes[j] && previous < address + sizes[i] {
        raise InvalidBinding
      }
    }
    addresses.push(address)
  }
  // Fixed disjoint scratch ownership: x[0..K), y[4096..4096+N).
  let operations = [
    @lunachip.Instruction::load(device, 0, addresses[1], self.columns),
    @lunachip.Instruction::bf16_gemv(
      device,
      4096,
      0,
      addresses[0],
      self.rows,
      self.columns,
    ),
    @lunachip.Instruction::store(device, 4096, addresses[2], self.rows),
  ] catch {
    e => raise Device(e)
  }
  @lunachip.PreparedProgram::new(operations) catch {
    e => raise Device(e)
  }
}
