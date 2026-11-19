///|
/// Bias-free BF16 dense SwiGLU with separate row-major weights. Ordered F32
/// projection sums round to BF16 before SiLU, after SiLU, after multiplication,
/// and after down projection. The product is shared by every output column.
/// Geometry/storage are pure; device scheduling and ownership are downstream.
pub struct DenseSwiGluPrecision {
  priv hidden : Int32
  priv intermediate : Int32
  priv rows : Int32
  priv weight_bytes : Int64
  priv product_bytes : Int64
  priv publication : ProjectionPublication
  priv partial_bytes : Int64
  priv dot_law : DenseDotLaw
}

///|
pub fn DenseSwiGluPrecision::new(
  hidden_width~ : Int32,
  intermediate_width~ : Int32,
  maximum_rows~ : Int32,
  publication~ : ProjectionPublication = LocalBFloat16,
  dot_law~ : DenseDotLaw = ScalarFloat32,
) -> Self raise PrecisionPlanError {
  guard hidden_width > 0 && intermediate_width > 0 && maximum_rows > 0 else {
    raise IllegalPlacement
  }
  let weight_bytes = expert_checked_mul(
    expert_checked_mul(hidden_width.to_int64(), intermediate_width.to_int64()),
    2L,
  )
  let product_bytes = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), intermediate_width.to_int64()),
    2L,
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), hidden_width.to_int64()),
      2L,
    ),
  )
  let partial_bytes = if publication == ReduceFloat32BeforeRound {
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), hidden_width.to_int64()),
      4L,
    )
  } else {
    0L
  }
  guard partial_bytes <= 9223372036854775807L - product_bytes else {
    raise ArithmeticOverflow
  }
  {
    hidden: hidden_width,
    intermediate: intermediate_width,
    rows: maximum_rows,
    weight_bytes,
    product_bytes,
    publication,
    partial_bytes,
    dot_law,
  }
}

///|
pub fn DenseSwiGluPrecision::geometry(self : Self) -> (Int32, Int32, Int32) {
  (self.hidden, self.intermediate, self.rows)
}

///|
pub fn DenseSwiGluPrecision::weight_bytes(self : Self) -> Int64 {
  self.weight_bytes
}

///|
pub fn DenseSwiGluPrecision::workspace_bytes(self : Self) -> Int64 {
  self.product_bytes + self.partial_bytes
}

///|
pub fn DenseSwiGluPrecision::product_bytes(self : Self) -> Int64 {
  self.product_bytes
}

///|
pub fn DenseSwiGluPrecision::publication(self : Self) -> ProjectionPublication {
  self.publication
}

///|
pub fn DenseSwiGluPrecision::dot_law(self : Self) -> DenseDotLaw {
  self.dot_law
}
