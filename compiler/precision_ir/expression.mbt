///|
/// Scalar storage boundaries. Arithmetic nodes operate on F32, never real
/// numbers with an implicit final cast. Rounding is nearest-even, with gradual
/// underflow. Device transcendental functions remain named primitives: their
/// implementation is NOT assumed correctly rounded or interchangeable.
pub(all) enum NumericScalar {
  Float32
  BFloat16
} derive(Eq)

///|
pub extend NumericScalar with Eq::{equal, not_equal}

///|
/// Preconditions of a program, not ranges inferred from a calibration sample.
pub(all) enum NumericDomain {
  AnyFloat
  FiniteFloat
  BoundedFinite(Double)
} derive(Eq)

///|
pub extend NumericDomain with Eq::{equal, not_equal}

///|
pub(all) enum NumericBinary {
  AddRN
  SubtractRN
  MultiplyRN
  DivideRN
  // Named device estimate, not algebraically interchangeable with DivideRN.
  // Its target-specific exceptional/denormal behavior is preserved by lowering.
  DivideEstimate
  MaximumNumber
  MinimumNumber
} derive(Eq)

///|
pub extend NumericBinary with Eq::{equal, not_equal}

///|
pub(all) enum NumericUnary {
  NaturalExp
  NaturalLogOnePlus
  ReciprocalSqrt
  SquareRoot
  Negate
} derive(Eq)

///|
pub extend NumericUnary with Eq::{equal, not_equal}

///|
/// Ordered F32 comparisons; NaN makes both false. Select evaluates only its
/// selected arm, not both arms followed by an arithmetic blend.
pub(all) enum NumericComparison {
  GreaterThan
  GreaterOrEqual
} derive(Eq)

///|
pub extend NumericComparison with Eq::{equal, not_equal}

///|
/// Pure expressions; memory reads are bound by the containing effect plan.
/// Constants retain their literal F32 bits, including signed zero and NaNs.
pub(all) enum NumericExpr {
  Input(Int)
  Float32Bits(UInt)
  Binary(NumericBinary, NumericExpr, NumericExpr)
  Unary(NumericUnary, NumericExpr)
  FusedMultiplyAdd(NumericExpr, NumericExpr, NumericExpr)
  Convert(NumericScalar, NumericExpr)
  Select(NumericComparison, NumericExpr, NumericExpr, NumericExpr, NumericExpr)
} derive(Eq)

///|
pub extend NumericExpr with Eq::{equal, not_equal}

///|
pub struct NumericExpression {
  priv inputs : ReadOnlyArray[(NumericScalar, NumericDomain)]
  priv root : NumericExpr
  priv output : NumericScalar
}

///|
fn NumericExpr::scalar(
  self : Self,
  inputs : ArrayView[(NumericScalar, NumericDomain)],
) -> NumericScalar raise PrecisionPlanError {
  match self {
    Input(index) => {
      guard index >= 0 && index < inputs.length() else {
        raise IllegalPlacement
      }
      inputs[index].0
    }
    Float32Bits(_) => Float32
    Convert(kind, value) => {
      ignore(value.scalar(inputs))
      kind
    }
    Unary(_, value) => {
      guard value.scalar(inputs) == Float32 else { raise IllegalPlacement }
      Float32
    }
    Binary(_, left, right) => {
      guard left.scalar(inputs) == Float32 && right.scalar(inputs) == Float32 else {
        raise IllegalPlacement
      }
      Float32
    }
    FusedMultiplyAdd(a, b, c) => {
      guard a.scalar(inputs) == Float32 &&
        b.scalar(inputs) == Float32 &&
        c.scalar(inputs) == Float32 else {
        raise IllegalPlacement
      }
      Float32
    }
    Select(_, a, b, yes, no) => {
      guard a.scalar(inputs) == Float32 && b.scalar(inputs) == Float32 else {
        raise IllegalPlacement
      }
      let result = yes.scalar(inputs)
      guard no.scalar(inputs) == result else { raise IllegalPlacement }
      result
    }
  }
}

///|
pub fn NumericExpression::new(
  inputs : ArrayView[(NumericScalar, NumericDomain)],
  root : NumericExpr,
) -> Self raise PrecisionPlanError {
  for (_, domain) in inputs {
    if domain is BoundedFinite(bound) {
      guard bound >= 0.0 && !bound.is_nan() && !bound.is_inf() else {
        raise IllegalPlacement
      }
    }
  }
  let output = root.scalar(inputs)
  { inputs: ReadOnlyArray::from_array(inputs), root, output, }
}

///|
pub fn NumericExpression::inputs(
  self : Self,
) -> ReadOnlyArray[(NumericScalar, NumericDomain)] {
  self.inputs
}

///|
pub fn NumericExpression::root(self : Self) -> NumericExpr {
  self.root
}

///|
pub fn NumericExpression::output(self : Self) -> NumericScalar {
  self.output
}
