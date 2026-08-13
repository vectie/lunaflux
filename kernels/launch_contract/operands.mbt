///|
/// Semantic address passed at one position in an AOT kernel parameter array.
/// There is deliberately no scalar or untyped payload variant.
pub(all) enum AotOperandRole {
  ExternalInput(@plan.ModelInputRef)
  WeightInput(@plan.TensorRef)
  ActivationInput(@plan.ActivationRef)
  RuntimeInput(@plan.OperationRuntimeInput)
  PersistentKvComponent(@plan.KvLayerId, @device_layout.DeviceKvComponent)
  ActivationOutput(@plan.ActivationRef)
  Workspace
} derive(Debug, Eq)

///|
pub struct AotOperand {
  role : AotOperandRole
  byte_count : Int64
  alignment : Int64
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
