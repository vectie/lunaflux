///|
pub struct LunaTileProgramInput {
  priv residual_add : LunaTileResidualAddSemantics?
  priv loop_extents : ReadOnlyArray[Int64]
  priv tensors : ReadOnlyArray[LunaTileTensorView]
  priv instructions : ReadOnlyArray[LunaTileInstruction]
  priv constraints : ReadOnlyArray[LunaTileConstraint]
} derive(Debug, Eq)

///|
pub fn LunaTileProgramInput::new(
  loop_extents~ : ArrayView[Int64],
  tensors~ : ArrayView[LunaTileTensorView],
  instructions~ : ArrayView[LunaTileInstruction],
  constraints~ : ArrayView[LunaTileConstraint],
) -> LunaTileProgramInput {
  {
    residual_add: None,
    loop_extents: ReadOnlyArray::from_array(loop_extents),
    tensors: ReadOnlyArray::from_array(tensors),
    instructions: ReadOnlyArray::from_array(instructions),
    constraints: ReadOnlyArray::from_array(constraints),
  }
}

///|
/// Constructs the only model-semantic LunaTile program currently admitted.
///
/// The ordered tensor IDs are abstract AOT operand roles. The constructor does
/// not accept a family label: exact residual-add structure is proven during
/// program admission before this opaque semantic contract can be published.
pub fn LunaTileProgramInput::residual_add(
  left~ : LunaTileTensorId,
  right~ : LunaTileTensorId,
  output~ : LunaTileTensorId,
  width~ : Int64,
  loop_extents~ : ArrayView[Int64],
  tensors~ : ArrayView[LunaTileTensorView],
  instructions~ : ArrayView[LunaTileInstruction],
  constraints~ : ArrayView[LunaTileConstraint],
) -> LunaTileProgramInput {
  {
    residual_add: Some({ left, right, output, width }),
    loop_extents: ReadOnlyArray::from_array(loop_extents),
    tensors: ReadOnlyArray::from_array(tensors),
    instructions: ReadOnlyArray::from_array(instructions),
    constraints: ReadOnlyArray::from_array(constraints),
  }
}

///|
/// Opaque, admission-proven ordered roles for exact BF16 residual addition.
pub struct LunaTileResidualAddSemantics {
  priv left : LunaTileTensorId
  priv right : LunaTileTensorId
  priv output : LunaTileTensorId
  priv width : Int64
} derive(Debug, Eq)

///|
pub fn LunaTileResidualAddSemantics::width(
  self : LunaTileResidualAddSemantics,
) -> Int64 {
  self.width
}

///|
pub struct LunaTileProgramDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileProgramDigest::from_sha256(
  value : String,
) -> LunaTileProgramDigest raise LunaTileError {
  if !luna_tile_is_sha256(value) {
    raise InvalidProgram(CanonicalEncoding)
  }
  LunaTileProgramDigest(value)
}

///|
pub fn LunaTileProgramDigest::as_hex(self : LunaTileProgramDigest) -> String {
  self.0
}

///|
pub struct LunaTileProgram {
  priv residual_add : LunaTileResidualAddSemantics?
  priv loop_extents : ReadOnlyArray[Int64]
  priv tensors : ReadOnlyArray[LunaTileTensorView]
  priv instructions : ReadOnlyArray[LunaTileInstruction]
  priv constraints : ReadOnlyArray[LunaTileConstraint]
  priv canonical : Bytes
  priv digest : LunaTileProgramDigest
} derive(Debug, Eq)

///|
pub fn LunaTileProgram::digest(self : LunaTileProgram) -> LunaTileProgramDigest {
  self.digest
}

///|
pub fn LunaTileProgram::residual_add_semantics(
  self : LunaTileProgram,
) -> LunaTileResidualAddSemantics? {
  self.residual_add
}

///|
pub fn LunaTileProgram::canonical_bytes(self : LunaTileProgram) -> Bytes {
  self.canonical
}

///|
pub fn LunaTileProgram::tensor_count(self : LunaTileProgram) -> Int {
  self.tensors.length()
}

///|
pub fn LunaTileProgram::uses_dtype(
  self : LunaTileProgram,
  dtype : LunaTileDType,
) -> Bool {
  for tensor in self.tensors {
    if tensor.dtype == dtype {
      return true
    }
  }
  false
}

///|
pub fn LunaTileProgram::instruction_count(self : LunaTileProgram) -> Int {
  self.instructions.length()
}

///|
pub fn LunaTileProgram::loop_rank(self : LunaTileProgram) -> Int {
  self.loop_extents.length()
}
