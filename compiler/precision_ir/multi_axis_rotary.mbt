///|
pub(all) enum MultiAxisRotaryExpression {
  Angle
  Cosine
  Sine
  Real
  Imaginary
}

///|
/// Packed BF16 heads with split-half rotation in a prefix. Coefficients and
/// each product round to BF16, then the sum rounds to BF16. The remaining
/// suffix is copied unchanged. Positions are three int32 axes per row.
/// Model builders supply the axis map and FP32 inverse frequency constants.
pub struct MultiAxisRotaryPrecision {
  priv heads : Int32
  priv width : Int32
  priv rotary : Int32
  priv rows : Int32
  priv positions : Int32
  priv axes : ReadOnlyArray[Int32]
  priv frequencies : ReadOnlyArray[Float]
}

///|
pub fn MultiAxisRotaryPrecision::new(
  head_count~ : Int32,
  head_width~ : Int32,
  rotary_width~ : Int32,
  maximum_rows~ : Int32,
  maximum_positions~ : Int32,
  axis_for_pair~ : ArrayView[Int32],
  inverse_frequencies~ : ArrayView[Float],
) -> Self raise PrecisionPlanError {
  guard head_count > 0 &&
    head_width > 0 &&
    rotary_width > 0 &&
    rotary_width % 2 == 0 &&
    rotary_width <= head_width &&
    maximum_rows > 0 &&
    maximum_positions > 0 &&
    axis_for_pair.length() == rotary_width / 2 &&
    inverse_frequencies.length() == rotary_width / 2 &&
    axis_for_pair.all(axis => axis >= 0 && axis < 3) &&
    inverse_frequencies.all(frequency => frequency > 0.0F && !frequency.is_inf()) else {
    raise IllegalPlacement
  }
  ignore(
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), head_count.to_int64()),
      head_width.to_int64() * 2L,
    ),
  )
  ignore(
    expert_checked_mul(maximum_rows.to_int64(), rotary_width.to_int64() * 2L),
  )
  {
    heads: head_count,
    width: head_width,
    rotary: rotary_width,
    rows: maximum_rows,
    positions: maximum_positions,
    axes: ReadOnlyArray::from_array(axis_for_pair),
    frequencies: ReadOnlyArray::from_array(inverse_frequencies),
  }
}

///|
pub fn MultiAxisRotaryPrecision::geometry(
  self : Self,
) -> (Int32, Int32, Int32, Int32, Int32) {
  (self.heads, self.width, self.rotary, self.rows, self.positions)
}

///|
pub fn MultiAxisRotaryPrecision::axes(self : Self) -> ReadOnlyArray[Int32] {
  self.axes
}

///|
pub fn MultiAxisRotaryPrecision::inverse_frequencies(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.frequencies
}

///|
pub fn MultiAxisRotaryPrecision::tensor_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * self.heads.to_int64() * self.width.to_int64() * 2L
}

///|
/// One cosine plane plus one sine plane, each [row,rotary_width/2] BF16.
pub fn MultiAxisRotaryPrecision::coefficient_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * self.rotary.to_int64() * 2L
}

///|
pub fn MultiAxisRotaryPrecision::expression(
  _self : Self,
  expression : MultiAxisRotaryExpression,
) -> NumericExpression raise PrecisionPlanError {
  let x = Input(0)
  let y = Input(1)
  let c = Input(2)
  let s = Input(3)
  let product = fn(left, right) {
    Convert(
      Float32,
      Convert(
        BFloat16,
        Binary(MultiplyRN, Convert(Float32, left), Convert(Float32, right)),
      ),
    )
  }
  match expression {
    Angle => numeric_expression([Float32, Float32], Binary(MultiplyRN, x, y))
    Cosine => numeric_expression([Float32], Convert(BFloat16, Unary(Cosine, x)))
    Sine => numeric_expression([Float32], Convert(BFloat16, Unary(Sine, x)))
    Real =>
      numeric_expression(
        [BFloat16, BFloat16, BFloat16, BFloat16],
        Convert(
          BFloat16,
          Binary(AddRN, product(x, c), Unary(Negate, product(y, s))),
        ),
      )
    Imaginary =>
      numeric_expression(
        [BFloat16, BFloat16, BFloat16, BFloat16],
        Convert(BFloat16, Binary(AddRN, product(y, c), product(x, s))),
      )
  }
}
