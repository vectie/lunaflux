///|
/// Learned stream collapse, independent of vocabulary projection or a model
/// name. Stable sigmoid branches, finite-value rejection and BF16 publication
/// are part of this operation; the scalar programs do not erase those guards.
pub struct LearnedStreamCollapsePrecision {
  priv streams : Int
  priv hidden : Int
  priv rows : Int
  priv epsilon : Double
  priv normalization_epsilon : Double
}

///|
pub fn LearnedStreamCollapsePrecision::new(
  stream_count~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
) -> Self raise PrecisionPlanError {
  guard stream_count > 1 &&
    stream_count <= 16 &&
    hidden_width > 0 &&
    hidden_width <= 65536 &&
    maximum_rows > 0 &&
    maximum_rows <= 1048576 &&
    control_epsilon > 0.0 &&
    !control_epsilon.is_inf() &&
    !control_epsilon.is_nan() &&
    normalization_epsilon > 0.0 &&
    !normalization_epsilon.is_inf() &&
    !normalization_epsilon.is_nan() else {
    raise IllegalPlacement
  }
  {
    streams: stream_count,
    hidden: hidden_width,
    rows: maximum_rows,
    epsilon: control_epsilon,
    normalization_epsilon,
  }
}

///|
pub fn LearnedTextIoPrecision::collapse_program(
  self : Self,
) -> LearnedStreamCollapsePrecision {
  // Enclosing constructor already establishes these geometry/epsilon bounds.
  {
    streams: self.streams,
    hidden: self.text.hidden,
    rows: self.text.rows,
    epsilon: self.control_epsilon,
    normalization_epsilon: self.control_normalization_epsilon,
  }
}

///|
pub fn LearnedStreamCollapsePrecision::geometry(self : Self) -> (Int, Int, Int) {
  (self.streams, self.hidden, self.rows)
}

///|
pub fn LearnedStreamCollapsePrecision::epsilons(
  self : Self,
) -> (Double, Double) {
  (self.epsilon, self.normalization_epsilon)
}

///|
pub(all) enum LearnedStreamExpression {
  SquareUpdate
  Mean
  Inverse
  DotUpdate
  Mix
  Affine
  SigmoidNonnegative
  Exponential
  SigmoidNegativeExponential
  Control
  CollapseUpdate
  Publish
}

///|
pub fn LearnedStreamCollapsePrecision::expression(
  _self : Self,
  operation : LearnedStreamExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x : NumericExpr = Input(0)
  let y : NumericExpr = Input(1)
  let z : NumericExpr = Input(2)
  let one = Float32Bits(0x3f800000U)
  let (types, root) : (Array[NumericScalar], NumericExpr) = match operation {
    SquareUpdate =>
      ([Float32, Float32], Binary(AddRN, x, Binary(MultiplyRN, y, y)))
    Mean => ([Float32, Float32], Binary(DivideRN, x, y))
    Inverse =>
      (
        [Float32, Float32],
        Binary(DivideRN, one, Unary(SquareRoot, Binary(AddRN, x, y))),
      )
    DotUpdate =>
      (
        [Float32, BFloat16, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, Convert(Float32, y), z)),
      )
    Mix => ([Float32, Float32], Binary(MultiplyRN, x, y))
    Affine =>
      ([Float32, Float32, Float32], Binary(AddRN, Binary(MultiplyRN, x, y), z))
    SigmoidNonnegative =>
      (
        [Float32],
        Binary(
          DivideRN,
          one,
          Binary(AddRN, one, Unary(NaturalExp, Unary(Negate, x))),
        ),
      )
    // The negative branch binds exp(value) once, then consumes it twice.
    SigmoidNegativeExponential =>
      ([Float32], Binary(DivideRN, x, Binary(AddRN, one, x)))
    Exponential => ([Float32], Unary(NaturalExp, x))
    Control => ([Float32, Float32], Binary(AddRN, x, y))
    CollapseUpdate =>
      ([Float32, Float32, Float32], Binary(AddRN, x, Binary(MultiplyRN, y, z)))
    Publish => ([Float32], Convert(BFloat16, x))
  }
  NumericExpression::new(types.map(kind => (kind, AnyFloat)), root)
}

///|
pub fn LearnedStreamCollapsePrecision::fold(
  self : Self,
  operation : LearnedStreamExpression,
) -> OrderedNumericFold raise PrecisionPlanError {
  OrderedNumericFold::new(
    extent=match operation {
      SquareUpdate | DotUpdate => self.hidden * self.streams
      CollapseUpdate => self.streams
      _ => raise IllegalPlacement
    },
    initial_bits=0U,
    update=self.expression(operation),
  )
}
