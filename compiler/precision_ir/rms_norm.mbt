///|
pub(all) enum NormalizationWeightLaw {
  DirectGamma
  OnePlusGamma
} derive(Debug, Eq)

///|
pub extend NormalizationWeightLaw with @debug.Debug::{to_repr}

///|
pub extend NormalizationWeightLaw with Eq::{equal, not_equal}

///|
pub(all) enum RmsNormMode {
  Weighted(NormalizationWeightLaw)
  Bf16NormBf16WeightSiluGate
} derive(Debug, Eq)

///|
pub extend RmsNormMode with @debug.Debug::{to_repr}

///|
pub extend RmsNormMode with Eq::{equal, not_equal}

///|
pub(all) enum RmsNormExpression {
  SquareUpdate
  Inverse
  Normalize
  Weight
  Gate
  Publish
}

///|
/// BF16 input/weight/output, explicit FP32 arithmetic and stage boundaries.
/// Input vectors may be views into packed projections; output vectors are packed.
/// This plan describes arithmetic and layout, not model identity or a device.
pub struct RmsNormPrecision {
  priv width : Int32
  priv vectors : Int32
  priv rows : Int32
  priv epsilon : Double
  priv mode : RmsNormMode
  priv vector_stride : Int32
  priv row_stride : Int64
}

///|
pub fn RmsNormPrecision::new(
  vector_width~ : Int32,
  vectors_per_row~ : Int32,
  maximum_rows~ : Int32,
  epsilon~ : Double,
  mode~ : RmsNormMode,
  input_vector_stride~ : Int32 = vector_width,
  input_row_stride? : Int64,
) -> RmsNormPrecision raise PrecisionPlanError {
  guard vector_width > 0 &&
    vectors_per_row > 0 &&
    maximum_rows > 0 &&
    input_vector_stride >= vector_width else {
    raise IllegalPlacement
  }
  guard epsilon > 0.0 && epsilon <= 1.0 && !epsilon.is_inf() else {
    raise MissingAccuracyContract
  }
  let row_stride = input_row_stride.unwrap_or(
    expert_checked_mul(
      vectors_per_row.to_int64(),
      input_vector_stride.to_int64(),
    ),
  )
  let span = expert_checked_mul(
    (vectors_per_row - 1).to_int64(),
    input_vector_stride.to_int64(),
  )
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
      expert_checked_mul(vector_width.to_int64(), 2L),
    ),
  )
  {
    width: vector_width,
    vectors: vectors_per_row,
    rows: maximum_rows,
    epsilon,
    mode,
    vector_stride: input_vector_stride,
    row_stride,
  }
}

///|
pub fn RmsNormPrecision::geometry(
  self : RmsNormPrecision,
) -> (Int32, Int32, Int32) {
  (self.width, self.vectors, self.rows)
}

///|
pub fn RmsNormPrecision::input_layout(
  self : RmsNormPrecision,
) -> (Int32, Int64) {
  (self.vector_stride, self.row_stride)
}

///|
pub fn RmsNormPrecision::epsilon(self : RmsNormPrecision) -> Double {
  self.epsilon
}

///|
pub fn RmsNormPrecision::mode(self : RmsNormPrecision) -> RmsNormMode {
  self.mode
}

///|
pub fn RmsNormPrecision::output_bytes(self : RmsNormPrecision) -> Int64 {
  self.rows.to_int64() * self.vectors.to_int64() * self.width.to_int64() * 2L
}

///|
pub fn RmsNormPrecision::expression(
  self : RmsNormPrecision,
  expression : RmsNormExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let z = Input(2)
  let one = Float32Bits(0x3f800000U)
  match expression {
    SquareUpdate =>
      numeric_expression(
        [Float32, Float32],
        Binary(AddRN, x, Binary(MultiplyRN, y, y)),
      )
    Inverse =>
      numeric_expression(
        [Float32, Float32, Float32],
        Unary(ReciprocalSqrt, Binary(AddRN, Binary(DivideRN, x, y), z)),
      )
    Normalize => {
      let product = Binary(MultiplyRN, x, y)
      numeric_expression(
        [Float32, Float32],
        if self.mode is Bf16NormBf16WeightSiluGate {
          Convert(Float32, Convert(BFloat16, product))
        } else {
          product
        },
      )
    }
    Weight => {
      let gamma = match self.mode {
        Weighted(OnePlusGamma) => Binary(AddRN, one, Convert(Float32, y))
        _ => Convert(Float32, y)
      }
      let product = Binary(MultiplyRN, x, gamma)
      numeric_expression(
        [Float32, BFloat16],
        if self.mode is Bf16NormBf16WeightSiluGate {
          Convert(Float32, Convert(BFloat16, product))
        } else {
          product
        },
      )
    }
    Gate => {
      guard self.mode is Bf16NormBf16WeightSiluGate else {
        raise IllegalPlacement
      }
      let gate = Convert(Float32, y)
      let silu = Binary(
        DivideRN,
        gate,
        Binary(AddRN, one, Unary(NaturalExp, Unary(Negate, gate))),
      )
      numeric_expression(
        [Float32, BFloat16],
        Convert(BFloat16, Binary(MultiplyRN, x, silu)),
      )
    }
    Publish => numeric_expression([Float32], Convert(BFloat16, x))
  }
}

///|
pub fn RmsNormPrecision::square_fold(
  self : RmsNormPrecision,
) -> OrderedNumericFold raise PrecisionPlanError {
  OrderedNumericFold::new(
    extent=self.width,
    initial_bits=0U,
    update=self.expression(SquareUpdate),
  )
}
