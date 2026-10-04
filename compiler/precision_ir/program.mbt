///|
/// Compute precision is independent of persistent storage precision.
pub(all) enum Accumulator {
  F32
} derive(Debug, Eq)

///|
pub(all) enum ComputeRoute {
  Native
  DecodeToBf16
  StartupBf16
} derive(Debug, Eq)

///|
pub struct LinearPrecision {
  priv activation : OperandPlan
  priv weight : OperandPlan
  priv output : OperandPlan
  priv route : ComputeRoute
  priv accumulator : Accumulator
} derive(Debug, Eq)

///|
/// Generic matrix semantic pass. Native low-bit compute requires a separate
/// advertised implementation; supporting a conversion is not that capability.
pub fn LinearPrecision::new(
  activation~ : OperandPlan,
  weight~ : OperandPlan,
  output~ : OperandPlan,
  route~ : ComputeRoute,
  accumulator~ : Accumulator,
) -> LinearPrecision raise PrecisionPlanError {
  if activation.role != Activation ||
    weight.role != Weight ||
    output.role != Output {
    raise IllegalPlacement
  }
  let (input_rows, input_columns) = activation.representation.shape()
  let (output_columns, weight_columns) = weight.representation.shape()
  if input_columns != weight_columns ||
    output.representation.shape() != (input_rows, output_columns) {
    raise IllegalPlacement
  }
  let weight_ok = match route {
    Native => weight.placement == NativeOperand
    DecodeToBf16 => weight.placement == DecodeInOperandTile
    StartupBf16 => weight.placement == StartupExpandBf16
  }
  if !weight_ok {
    raise IllegalPlacement
  }
  if route != Native &&
    (
      activation.representation.element() != Bf16 ||
      activation.placement != NativeOperand ||
      output.placement != NativeOperand ||
      output.representation.element() != Bf16
    ) {
    raise NoImplementation
  }
  { activation, weight, output, route, accumulator, }
}

///|
pub fn LinearPrecision::operands(
  self : LinearPrecision,
) -> (OperandPlan, OperandPlan, OperandPlan) {
  (self.activation, self.weight, self.output)
}

///|
pub fn LinearPrecision::route(self : LinearPrecision) -> ComputeRoute {
  self.route
}

///|
pub fn LinearPrecision::accumulator(self : LinearPrecision) -> Accumulator {
  self.accumulator
}
