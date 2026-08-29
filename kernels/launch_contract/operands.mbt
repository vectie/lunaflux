///|
const ABSOLUTE_MAX_OPERANDS_PER_CONTRACT : Int = 32

///|
/// Semantic address passed at one position in an AOT kernel parameter array.
/// There is deliberately no scalar or untyped payload variant.
pub(all) enum AotOperandRole {
  ExternalInput(@plan.ModelInputRef)
  WeightInput(@plan.TensorRef)
  WeightScaleInput(@plan.TensorRef, @plan.ScaleTensorRef)
  ActivationInput(@plan.ActivationRef)
  RuntimeInput(@plan.OperationRuntimeInput)
  PersistentKvComponent(@plan.KvLayerId, @device_layout.DeviceKvComponent)
  ActivationOutput(@plan.ActivationRef)
  Workspace
} derive(Debug, Eq)

///|
pub struct AotOperand {
  priv role : AotOperandRole
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
pub fn AotOperand::new(
  role~ : AotOperandRole,
  byte_count~ : Int64,
  alignment~ : Int64,
) -> AotOperand raise LaunchContractError {
  if byte_count <= 0L {
    raise InvalidOperandDefinition(NonPositiveByteCount)
  }
  if alignment <= 0L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L {
    raise InvalidOperandDefinition(InvalidAlignment)
  }
  { role, byte_count, alignment }
}

///|
pub fn AotOperand::role(self : AotOperand) -> AotOperandRole {
  self.role
}

///|
pub fn AotOperand::byte_count(self : AotOperand) -> Int64 {
  self.byte_count
}

///|
pub fn AotOperand::alignment(self : AotOperand) -> Int64 {
  self.alignment
}

///|
fn checked_owned_operands(
  operands : ArrayView[AotOperand],
  operation : @plan.OperationId,
) -> ReadOnlyArray[AotOperand] raise LaunchContractError {
  if operands.is_empty() {
    raise InvalidOperandDefinition(MissingSemanticRole)
  }
  if operands.length() > ABSOLUTE_MAX_OPERANDS_PER_CONTRACT {
    raise OperandCountExceeded(
      operation~,
      maximum=ABSOLUTE_MAX_OPERANDS_PER_CONTRACT,
    )
  }
  ReadOnlyArray::from_array(operands)
}
