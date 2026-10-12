///|
pub(all) enum DenseSwiGluExpression {
  ProjectionUpdate
  ProjectionPublish
  Activation
  Product
}

///|
pub fn DenseSwiGluPrecision::expression(
  self : Self,
  operation : DenseSwiGluExpression,
) -> NumericExpression raise PrecisionPlanError {
  ignore(self)
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  match operation {
    ProjectionUpdate =>
      numeric_expression(
        [Float32, BFloat16, BFloat16],
        Binary(
          AddRN,
          x,
          Binary(MultiplyRN, Convert(Float32, y), Convert(Float32, z)),
        ),
      )
    ProjectionPublish => numeric_expression([Float32], Convert(BFloat16, x))
    Activation => {
      let g = Convert(Float32, x)
      numeric_expression(
        [BFloat16],
        Convert(
          BFloat16,
          Binary(
            DivideRN,
            g,
            Binary(
              AddRN,
              Float32Bits(0x3f800000U),
              Unary(NaturalExp, Unary(Negate, g)),
            ),
          ),
        ),
      )
    }
    Product =>
      numeric_expression(
        [BFloat16, BFloat16],
        Convert(
          BFloat16,
          Binary(MultiplyRN, Convert(Float32, x), Convert(Float32, y)),
        ),
      )
  }
}

///|
pub fn DenseSwiGluPrecision::projection_fold(
  self : Self,
  down~ : Bool,
) -> OrderedNumericFold raise PrecisionPlanError {
  OrderedNumericFold::new(
    extent=if down { self.intermediate } else { self.hidden },
    initial_bits=0U,
    update=self.expression(ProjectionUpdate),
  )
}

///|
/// Counts, input, gate/up/down weights, product workspace, output. Product is
/// uninitialized until stage zero; the compiler owns this producer dependency.
pub fn DenseSwiGluPrecision::program(
  self : Self,
) -> @parallel.Program raise PrecisionPlanError {
  let frame = self.rows.to_int64() * self.hidden.to_int64() * 2L
  numeric_effects(
    [
      20L,
      frame,
      self.weight_bytes,
      self.weight_bytes,
      self.weight_bytes,
      self.product_bytes,
      frame,
    ],
    [true, true, true, true, true, false, false],
    [
      [(0, Read), (1, Read), (2, Read), (3, Read), (5, Write)],
      [(0, Read), (4, Read), (5, Read), (6, Write)],
    ],
  )
}
