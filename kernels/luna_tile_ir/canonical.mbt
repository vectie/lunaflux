///|
priv struct LunaTileEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn LunaTileEncoder::append_byte(
  self : LunaTileEncoder,
  value : Byte,
) -> Unit raise LunaTileError {
  if self.bytes.length() >= self.maximum {
    raise LimitExceeded(CanonicalBytes)
  }
  self.bytes.push(value)
}

///|
fn LunaTileEncoder::append_u64(
  self : LunaTileEncoder,
  value : UInt64,
) -> Unit raise LunaTileError {
  for byte in value.to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn LunaTileEncoder::append_int(
  self : LunaTileEncoder,
  value : Int,
) -> Unit raise LunaTileError {
  self.append_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn LunaTileEncoder::append_i64(
  self : LunaTileEncoder,
  value : Int64,
) -> Unit raise LunaTileError {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn luna_tile_dtype_code(value : LunaTileDType) -> Int {
  match value {
    BFloat16 => 1
    Float16 => 2
    Float32 => 3
    Int8 => 4
    Int32 => 5
  }
}

///|
fn luna_tile_space_code(value : LunaTileAddressSpace) -> Int {
  match value {
    Global => 1
    Shared => 2
    Register => 3
  }
}

///|
fn LunaTileEncoder::append_affine(
  self : LunaTileEncoder,
  value : LunaTileAffineExpr,
) -> Unit raise LunaTileError {
  self.append_i64(value.constant)
  self.append_int(value.terms.length())
  for term in value.terms {
    self.append_int(term.loop_dimension)
    self.append_i64(term.coefficient)
  }
}

///|
fn LunaTileEncoder::append_region(
  self : LunaTileEncoder,
  value : LunaTileRegion,
) -> Unit raise LunaTileError {
  self.append_int(value.tensor.as_int())
  self.append_int(value.origins.length())
  for origin in value.origins {
    self.append_affine(origin)
  }
  for extent in value.extents {
    self.append_i64(extent)
  }
}

///|
fn LunaTileEncoder::append_instruction(
  self : LunaTileEncoder,
  instruction : LunaTileInstruction,
) -> Unit raise LunaTileError {
  match instruction {
    LunaTileCopy(source, destination) => {
      self.append_int(1)
      self.append_region(source)
      self.append_region(destination)
    }
    LunaTileAsyncCopy(source, destination, stage) => {
      self.append_int(2)
      self.append_region(source)
      self.append_region(destination)
      self.append_int(stage.as_int())
    }
    LunaTileBarrier(stage, scope) => {
      self.append_int(3)
      self.append_int(stage.as_int())
      self.append_int(
        match scope {
          Warp => 1
          Block => 2
        },
      )
    }
    LunaTilePipelineStage(stage) => {
      self.append_int(4)
      self.append_int(stage.as_int())
    }
    LunaTileMma(destination~, left~, right~, accumulator~) => {
      self.append_int(5)
      self.append_int(destination.as_int())
      self.append_int(left.as_int())
      self.append_int(right.as_int())
      self.append_int(accumulator.as_int())
    }
    LunaTileReduce(kind, destination~, source~, axis~) => {
      self.append_int(6)
      self.append_int(
        match kind {
          Sum => 1
          Maximum => 2
        },
      )
      self.append_int(destination.as_int())
      self.append_int(source.as_int())
      self.append_int(axis)
    }
    LunaTileElementwise(kind, destination~, inputs~) => {
      self.append_int(7)
      self.append_int(
        match kind {
          Add => 1
          Multiply => 2
          Maximum => 3
          Silu => 4
        },
      )
      self.append_int(destination.as_int())
      self.append_int(inputs.length())
      for input in inputs {
        self.append_int(input.as_int())
      }
    }
  }
}

///|
fn LunaTileEncoder::append_constraint(
  self : LunaTileEncoder,
  constraint : LunaTileConstraint,
) -> Unit raise LunaTileError {
  match constraint {
    LunaTileDimensionEquals(id, tensor, dimension, value) => {
      self.append_int(1)
      self.append_int(id.as_int())
      self.append_int(tensor.as_int())
      self.append_int(dimension)
      self.append_i64(value)
    }
    LunaTileDimensionMultipleOf(id, tensor, dimension, value) => {
      self.append_int(2)
      self.append_int(id.as_int())
      self.append_int(tensor.as_int())
      self.append_int(dimension)
      self.append_i64(value)
    }
    LunaTileAlignmentAtLeast(id, tensor, value) => {
      self.append_int(3)
      self.append_int(id.as_int())
      self.append_int(tensor.as_int())
      self.append_i64(value)
    }
  }
}

///|
fn encode_luna_tile_program(
  residual_add : LunaTileResidualAddSemantics?,
  loop_extents : ReadOnlyArray[Int64],
  tensors : ReadOnlyArray[LunaTileTensorView],
  instructions : ReadOnlyArray[LunaTileInstruction],
  constraints : ReadOnlyArray[LunaTileConstraint],
  maximum : Int,
) -> Bytes raise LunaTileError {
  let encoder : LunaTileEncoder = { bytes: [], maximum }
  for byte in @utf8.encode("LUNATILE-IR-V2") {
    encoder.append_byte(byte)
  }
  match residual_add {
    None => {
      encoder.append_int(0)
      encoder.append_int(0)
      encoder.append_int(0)
      encoder.append_int(0)
      encoder.append_i64(0L)
    }
    Some(semantic) => {
      encoder.append_int(1)
      encoder.append_int(semantic.left.as_int())
      encoder.append_int(semantic.right.as_int())
      encoder.append_int(semantic.output.as_int())
      encoder.append_i64(semantic.width)
    }
  }
  encoder.append_int(loop_extents.length())
  for extent in loop_extents {
    encoder.append_i64(extent)
  }
  encoder.append_int(tensors.length())
  for tensor in tensors {
    encoder.append_int(tensor.id.as_int())
    encoder.append_int(luna_tile_dtype_code(tensor.dtype))
    encoder.append_int(luna_tile_space_code(tensor.address_space))
    encoder.append_i64(tensor.offset_bytes)
    encoder.append_i64(tensor.alignment_bytes)
    encoder.append_int(tensor.shape.length())
    for dimension in tensor.shape {
      encoder.append_i64(dimension)
    }
    for stride in tensor.strides_bytes {
      encoder.append_i64(stride)
    }
  }
  encoder.append_int(instructions.length())
  for instruction in instructions {
    encoder.append_instruction(instruction)
  }
  encoder.append_int(constraints.length())
  for constraint in constraints {
    encoder.append_constraint(constraint)
  }
  Bytes::from_array(encoder.bytes)
}
