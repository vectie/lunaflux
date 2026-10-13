///|
/// Explicit numerical alternatives. Chunk evaluation reassociates recurrence
/// into a causal triangular system; it is not an exact F32 rewrite.
pub(all) enum RecurrentEvaluationLaw {
  OrderedTokenFloat32
  ChunkTriangularFloat32
} derive(Eq, Debug)

///|
pub extend RecurrentEvaluationLaw with Eq::{equal, not_equal}

///|
pub extend RecurrentEvaluationLaw with @debug.Debug::{to_repr}

///|
pub(all) enum RecurrentChunkExpression {
  GatePrefix
  DecayedOperand
  ProductUpdate
  SubtractProduct
  Scale
}

///|
pub fn RecurrentChunkExpression::program(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match self {
    GatePrefix => numeric_expression([Float32, Float32], Binary(AddRN, x, y))
    DecayedOperand =>
      numeric_expression(
        [Float32, Float32],
        Binary(MultiplyRN, x, Unary(NaturalExp, y)),
      )
    ProductUpdate =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, y, z)),
      )
    SubtractProduct =>
      numeric_expression(
        [Float32, Float32, Float32],
        Binary(SubtractRN, x, Binary(MultiplyRN, y, z)),
      )
    Scale => numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
  }
}
