///|
pub struct LunaTileTensorId(Int) derive(Debug, Eq, Hash)

///|
pub struct LunaTileStageId(Int) derive(Debug, Eq, Hash)

///|
pub struct LunaTileConstraintId(Int) derive(Debug, Eq, Hash)

///|
/// Stable, source-safe identity used to derive a CUDA entry-point symbol.
/// Caller text is never accepted as source or as a symbol fragment.
pub struct LunaTileEntryPointId(Int) derive(Debug, Eq, Hash)

///|
pub fn LunaTileTensorId::from_stable_int(
  value : Int,
) -> LunaTileTensorId raise LunaTileError {
  if value <= 0 {
    raise InvalidProgram(Identifier)
  }
  LunaTileTensorId(value)
}

///|
pub fn LunaTileTensorId::as_int(self : LunaTileTensorId) -> Int {
  self.0
}

///|
pub fn LunaTileStageId::from_stable_int(
  value : Int,
) -> LunaTileStageId raise LunaTileError {
  if value <= 0 {
    raise InvalidProgram(Identifier)
  }
  LunaTileStageId(value)
}

///|
pub fn LunaTileStageId::as_int(self : LunaTileStageId) -> Int {
  self.0
}

///|
pub fn LunaTileConstraintId::from_stable_int(
  value : Int,
) -> LunaTileConstraintId raise LunaTileError {
  if value <= 0 {
    raise InvalidProgram(Identifier)
  }
  LunaTileConstraintId(value)
}

///|
pub fn LunaTileConstraintId::as_int(self : LunaTileConstraintId) -> Int {
  self.0
}

///|
pub fn LunaTileEntryPointId::from_stable_int(
  value : Int,
) -> LunaTileEntryPointId raise LunaTileError {
  if value <= 0 {
    raise InvalidProgram(Identifier)
  }
  LunaTileEntryPointId(value)
}

///|
pub fn LunaTileEntryPointId::as_int(self : LunaTileEntryPointId) -> Int {
  self.0
}
