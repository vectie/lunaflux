///|
fn luna_tile_tensor(
  tensors : ReadOnlyArray[LunaTileTensorView],
  id : LunaTileTensorId,
) -> LunaTileTensorView? {
  for tensor in tensors {
    if tensor.id == id {
      return Some(tensor)
    }
  }
  None
}

///|
fn luna_tile_is_power_of_two(value : Int64) -> Bool {
  value > 0L && (value & (value - 1L)) == 0L
}

///|
fn luna_tile_checked_add(
  left : Int64,
  right : Int64,
  rule : LunaTileValidationRule,
) -> Int64 raise LunaTileError {
  if left < 0L || right < 0L || left > 9223372036854775807L - right {
    raise SizeOverflow(rule)
  }
  left + right
}

///|
fn luna_tile_checked_mul(
  left : Int64,
  right : Int64,
  rule : LunaTileValidationRule,
) -> Int64 raise LunaTileError {
  if left < 0L ||
    right < 0L ||
    (left != 0L && right > 9223372036854775807L / left) {
    raise SizeOverflow(rule)
  }
  left * right
}

///|
fn validate_luna_tile_tensor(
  tensor : LunaTileTensorView,
  limits : LunaTileLimits,
) -> Unit raise LunaTileError {
  let rank = tensor.shape.length()
  if rank <= 0 ||
    rank > limits.max_rank ||
    tensor.strides_bytes.length() != rank {
    raise InvalidTensor(tensor.id, TensorShape)
  }
  if tensor.offset_bytes < 0L {
    raise InvalidTensor(tensor.id, TensorOffset)
  }
  if !luna_tile_is_power_of_two(tensor.alignment_bytes) ||
    tensor.offset_bytes % tensor.alignment_bytes != 0L {
    raise InvalidTensor(tensor.id, TensorAlignment)
  }
  let element_bytes = luna_tile_dtype_bytes(tensor.dtype)
  let mut maximum_offset = tensor.offset_bytes
  for index in 0..<rank {
    let extent = tensor.shape[index]
    let stride = tensor.strides_bytes[index]
    if extent <= 0L {
      raise InvalidTensor(tensor.id, TensorShape)
    }
    if stride < element_bytes || stride % element_bytes != 0L {
      raise InvalidTensor(tensor.id, TensorStride)
    }
    maximum_offset = luna_tile_checked_add(
      maximum_offset,
      luna_tile_checked_mul(extent - 1L, stride, TensorExtent),
      TensorExtent,
    )
  }
  ignore(luna_tile_checked_add(maximum_offset, element_bytes, TensorExtent))
}

///|
fn validate_luna_tile_affine(
  expression : LunaTileAffineExpr,
  loop_extents : ReadOnlyArray[Int64],
) -> Int64 raise LunaTileError {
  if expression.constant < 0L {
    raise InvalidProgram(AffineExpression)
  }
  let mut maximum = expression.constant
  let mut previous = -1
  for term in expression.terms {
    if term.loop_dimension <= previous ||
      term.loop_dimension < 0 ||
      term.loop_dimension >= loop_extents.length() ||
      term.coefficient < 0L {
      raise InvalidProgram(AffineExpression)
    }
    previous = term.loop_dimension
    maximum = luna_tile_checked_add(
      maximum,
      luna_tile_checked_mul(
        loop_extents[term.loop_dimension] - 1L,
        term.coefficient,
        AffineExpression,
      ),
      AffineExpression,
    )
  }
  maximum
}

///|
fn validate_luna_tile_region(
  region : LunaTileRegion,
  tensors : ReadOnlyArray[LunaTileTensorView],
  loop_extents : ReadOnlyArray[Int64],
) -> LunaTileTensorView raise LunaTileError {
  guard luna_tile_tensor(tensors, region.tensor) is Some(tensor) else {
    raise InvalidTensor(region.tensor, Identifier)
  }
  if region.origins.length() != tensor.shape.length() ||
    region.extents.length() != tensor.shape.length() {
    raise InvalidTensor(region.tensor, RegionRank)
  }
  for index in 0..<tensor.shape.length() {
    let extent = region.extents[index]
    if extent <= 0L {
      raise InvalidTensor(region.tensor, RegionBounds)
    }
    let maximum_origin = validate_luna_tile_affine(
      region.origins[index],
      loop_extents,
    )
    if maximum_origin > tensor.shape[index] - extent {
      raise InvalidTensor(region.tensor, RegionBounds)
    }
  }
  tensor
}

///|
fn same_luna_tile_region_shape(
  left : LunaTileRegion,
  right : LunaTileRegion,
) -> Bool {
  left.extents == right.extents
}

///|
fn validate_luna_tile_copy(
  source : LunaTileRegion,
  destination : LunaTileRegion,
  tensors : ReadOnlyArray[LunaTileTensorView],
  loop_extents : ReadOnlyArray[Int64],
  asynchronous : Bool,
) -> Unit raise LunaTileError {
  let source_tensor = validate_luna_tile_region(source, tensors, loop_extents)
  let destination_tensor = validate_luna_tile_region(
    destination, tensors, loop_extents,
  )
  if source_tensor.dtype != destination_tensor.dtype {
    raise InvalidProgram(CopyDType)
  }
  if !same_luna_tile_region_shape(source, destination) {
    raise InvalidProgram(CopyShape)
  }
  if asynchronous {
    if source_tensor.address_space != Global ||
      destination_tensor.address_space != Shared {
      raise InvalidProgram(CopyAddressSpace)
    }
  } else if source_tensor.address_space == Register &&
    destination_tensor.address_space == Register {
    raise InvalidProgram(CopyAddressSpace)
  }
}

///|
fn require_luna_tile_tensor(
  tensors : ReadOnlyArray[LunaTileTensorView],
  id : LunaTileTensorId,
) -> LunaTileTensorView raise LunaTileError {
  guard luna_tile_tensor(tensors, id) is Some(tensor) else {
    raise InvalidTensor(id, Identifier)
  }
  tensor
}

///|
fn validate_luna_tile_mma(
  tensors : ReadOnlyArray[LunaTileTensorView],
  destination : LunaTileTensorId,
  left : LunaTileTensorId,
  right : LunaTileTensorId,
  accumulator : LunaTileTensorId,
) -> Unit raise LunaTileError {
  let output = require_luna_tile_tensor(tensors, destination)
  let lhs = require_luna_tile_tensor(tensors, left)
  let rhs = require_luna_tile_tensor(tensors, right)
  let acc = require_luna_tile_tensor(tensors, accumulator)
  if output.address_space != Register ||
    acc.address_space != Register ||
    lhs.address_space == Global ||
    rhs.address_space == Global {
    raise InvalidProgram(OperationAddressSpace)
  }
  if lhs.shape.length() != 2 ||
    rhs.shape.length() != 2 ||
    output.shape.length() != 2 ||
    acc.shape != output.shape ||
    lhs.shape[0] != output.shape[0] ||
    rhs.shape[1] != output.shape[1] ||
    lhs.shape[1] != rhs.shape[0] {
    raise InvalidProgram(MmaShape)
  }
  if !((lhs.dtype == BFloat16 || lhs.dtype == Float16) &&
    rhs.dtype == lhs.dtype &&
    output.dtype == Float32 &&
    acc.dtype == Float32) {
    raise InvalidProgram(MmaDType)
  }
}

///|
fn validate_luna_tile_reduce(
  tensors : ReadOnlyArray[LunaTileTensorView],
  destination : LunaTileTensorId,
  source : LunaTileTensorId,
  axis : Int,
) -> Unit raise LunaTileError {
  let output = require_luna_tile_tensor(tensors, destination)
  let input = require_luna_tile_tensor(tensors, source)
  if output.address_space != Register || input.address_space != Register {
    raise InvalidProgram(OperationAddressSpace)
  }
  if output == input || !(input.dtype is (BFloat16 | Float16 | Float32)) {
    raise InvalidProgram(Reduction)
  }
  if axis < 0 ||
    axis >= input.shape.length() ||
    output.shape.length() != input.shape.length() ||
    output.dtype != input.dtype {
    raise InvalidProgram(Reduction)
  }
  for index in 0..<input.shape.length() {
    let expected = if index == axis { 1L } else { input.shape[index] }
    if output.shape[index] != expected {
      raise InvalidProgram(Reduction)
    }
  }
}

///|
fn validate_luna_tile_elementwise(
  tensors : ReadOnlyArray[LunaTileTensorView],
  kind : LunaTileElementwiseKind,
  destination : LunaTileTensorId,
  inputs : ReadOnlyArray[LunaTileTensorId],
) -> Unit raise LunaTileError {
  let output = require_luna_tile_tensor(tensors, destination)
  if output.address_space != Register {
    raise InvalidProgram(OperationAddressSpace)
  }
  let expected_inputs = match kind {
    Add | Multiply | Maximum => 2
    Silu => 1
  }
  if inputs.length() != expected_inputs ||
    !(output.dtype is (BFloat16 | Float16 | Float32)) {
    raise InvalidProgram(Elementwise)
  }
  for id in inputs {
    let input = require_luna_tile_tensor(tensors, id)
    if input.address_space != Register {
      raise InvalidProgram(OperationAddressSpace)
    }
    if input.shape != output.shape || input.dtype != output.dtype {
      raise InvalidProgram(Elementwise)
    }
  }
}
