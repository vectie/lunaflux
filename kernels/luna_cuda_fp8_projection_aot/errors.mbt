///|
pub(all) enum Fp8ProjectionAotRule {
  Limits
  ModelIdentity
  Operation
  OperationKind
  NumericExecution
  ActivationScalePolicy
  CompoundGatedActivationScale
  NumericSchema
  DeviceTarget
  CompilerPolicy
  Profile
  OperandOrder
  OperandSize
  OperandAlignment
  Source
  Recipe
  FinalBinding
} derive(Debug, Eq)

///|
pub(all) suberror Fp8ProjectionAotError {
  Invalid(Fp8ProjectionAotRule)
  Unsupported(Fp8ProjectionAotRule)
  Mismatch(Fp8ProjectionAotRule)
  LimitExceeded(rule~ : Fp8ProjectionAotRule, maximum~ : Int)
} derive(Debug, Eq)
