///|
pub(all) enum Bf16GateExpression {
  Sigmoid
  Product
}

///|
/// A shared BF16 sigmoid/product law. Recurrent SiLU-gated normalization uses
/// different FP32 gate stages and deliberately does not consume this law.
pub fn Bf16GateExpression::program(
  self : Self,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  match self {
    Sigmoid =>
      numeric_expression(
        [BFloat16],
        Convert(
          BFloat16,
          Binary(
            DivideRN,
            Float32Bits(0x3f800000U),
            Binary(
              AddRN,
              Float32Bits(0x3f800000U),
              Unary(NaturalExp, Unary(Negate, Convert(Float32, x))),
            ),
          ),
        ),
      )
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
/// Packed BF16 values/output, with a possibly strided BF16 gate view. This
/// captures output gates embedded in head-interleaved Q/gate projections.
pub struct PointwiseGatePrecision {
  priv width : Int32
  priv vectors : Int32
  priv rows : Int32
  priv gate_vector_stride : Int32
  priv gate_row_stride : Int64
  priv gate_offset : Int32
}

///|
pub fn PointwiseGatePrecision::new(
  vector_width~ : Int32,
  vectors_per_row~ : Int32,
  maximum_rows~ : Int32,
  gate_vector_stride~ : Int32 = vector_width,
  gate_row_stride? : Int64,
  gate_element_offset~ : Int32 = 0,
) -> Self raise PrecisionPlanError {
  guard vector_width > 0 &&
    vectors_per_row > 0 &&
    maximum_rows > 0 &&
    gate_vector_stride >= vector_width &&
    gate_element_offset >= 0 else {
    raise IllegalPlacement
  }
  let row_stride = gate_row_stride.unwrap_or(
    expert_checked_mul(
      vectors_per_row.to_int64(),
      gate_vector_stride.to_int64(),
    ),
  )
  let span = expert_checked_mul(
      (vectors_per_row - 1).to_int64(),
      gate_vector_stride.to_int64(),
    ) +
    gate_element_offset.to_int64()
  guard row_stride >= span && row_stride - span >= vector_width.to_int64() else {
    raise IllegalPlacement
  }
  ignore(
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), row_stride),
      2L,
    ),
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), vectors_per_row.to_int64()),
      vector_width.to_int64() * 2L,
    ),
  )
  {
    width: vector_width,
    vectors: vectors_per_row,
    rows: maximum_rows,
    gate_vector_stride,
    gate_row_stride: row_stride,
    gate_offset: gate_element_offset,
  }
}

///|
pub fn PointwiseGatePrecision::geometry(self : Self) -> (Int32, Int32, Int32) {
  (self.width, self.vectors, self.rows)
}

///|
pub fn PointwiseGatePrecision::gate_layout(
  self : Self,
) -> (Int32, Int64, Int32) {
  (self.gate_vector_stride, self.gate_row_stride, self.gate_offset)
}

///|
pub fn PointwiseGatePrecision::output_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * self.vectors.to_int64() * self.width.to_int64() * 2L
}
