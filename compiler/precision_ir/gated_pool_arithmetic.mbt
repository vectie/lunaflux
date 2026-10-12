///|
/// Named scalar boundaries of the generic learned pool. Loop order is part of
/// the program: all reductions are ascending, seeded with +0 (maximum -inf).
pub(all) enum GatedPoolExpression {
  ProjectionUpdate
  ScoreBias
  Maximum
  DenominatorUpdate
  WeightedUpdate
  Publish
  SquareUpdate
  Inverse
  Normalize
}

///|
/// Replaces one declared numerical program with a DIFFERENT explicit program;
/// it does not claim FMA is an equivalent optimization of separate rounding.
pub fn LearnedGatedPoolPrecision::with_projection_multiply_add(
  self : Self,
  law : Float32AffineLaw,
) -> Self {
  { ..self, projection_multiply_add: law, }
}

///|
pub fn LearnedGatedPoolPrecision::expression(
  self : Self,
  operation : GatedPoolExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x : NumericExpr = Input(0)
  let y : NumericExpr = Input(1)
  let z : NumericExpr = Input(2)
  let (types, root) : (Array[NumericScalar], NumericExpr) = match operation {
    ProjectionUpdate =>
      (
        [Float32, Float32, BFloat16],
        match self.projection_multiply_add {
          SeparateMultiplyAdd =>
            Binary(AddRN, x, Binary(MultiplyRN, y, Convert(Float32, z)))
          FusedMultiplyAdd => FusedMultiplyAdd(y, Convert(Float32, z), x)
        },
      )
    ScoreBias => ([Float32, Float32], Binary(AddRN, x, y))
    Maximum => ([Float32, Float32], Binary(MaximumNumber, x, y))
    DenominatorUpdate =>
      (
        [Float32, Float32, Float32],
        Binary(AddRN, x, Unary(NaturalExp, Binary(SubtractRN, y, z))),
      )
    WeightedUpdate =>
      (
        [Float32, Float32, Float32, Float32, Float32],
        Binary(
          AddRN,
          x,
          Binary(
            MultiplyRN,
            y,
            Binary(
              DivideRN,
              Unary(NaturalExp, Binary(SubtractRN, z, Input(3))),
              Input(4),
            ),
          ),
        ),
      )
    Publish => ([Float32], Convert(BFloat16, x))
    SquareUpdate =>
      ([Float32, Float32], Binary(AddRN, x, Binary(MultiplyRN, y, y)))
    Inverse =>
      (
        [Float32, Float32, Float32],
        Unary(ReciprocalSqrt, Binary(AddRN, Binary(DivideRN, x, y), z)),
      )
    Normalize =>
      (
        [BFloat16, Float32, BFloat16],
        Convert(
          BFloat16,
          Binary(
            MultiplyRN,
            Binary(MultiplyRN, Convert(Float32, x), y),
            Convert(Float32, z),
          ),
        ),
      )
  }
  NumericExpression::new(types.map(kind => (kind, AnyFloat)), root)
}

///|
/// Ordered loop extent, separate from physical thread/block dimensions.
pub fn LearnedGatedPoolPrecision::reduction_extent(
  self : Self,
  operation : GatedPoolExpression,
) -> Int raise PrecisionPlanError {
  match operation {
    ProjectionUpdate => self.input_width
    Maximum | DenominatorUpdate | WeightedUpdate =>
      (if self.overlap { 2 } else { 1 }) * self.ratio
    SquareUpdate => self.feature_width
    _ => raise IllegalPlacement
  }
}
