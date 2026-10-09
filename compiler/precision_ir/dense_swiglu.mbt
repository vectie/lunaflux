///|
/// Bias-free BF16 dense SwiGLU with separate row-major weights. Ordered F32
/// projection sums round to BF16 before SiLU, after SiLU, after multiplication,
/// and after down projection. The product is shared by every output column.
/// Geometry/storage are pure; device scheduling and ownership are downstream.
pub struct DenseSwiGluPrecision {
  priv hidden : Int
  priv intermediate : Int
  priv rows : Int
  priv weight_bytes : Int64
  priv product_bytes : Int64
}

///|
pub fn DenseSwiGluPrecision::new(
  hidden_width~ : Int,
  intermediate_width~ : Int,
  maximum_rows~ : Int,
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
  {
    hidden: hidden_width,
    intermediate: intermediate_width,
    rows: maximum_rows,
    weight_bytes,
    product_bytes,
  }
}

///|
pub fn DenseSwiGluPrecision::geometry(self : Self) -> (Int, Int, Int) {
  (self.hidden, self.intermediate, self.rows)
}

///|
pub fn DenseSwiGluPrecision::weight_bytes(self : Self) -> Int64 {
  self.weight_bytes
}

///|
pub fn DenseSwiGluPrecision::workspace_bytes(self : Self) -> Int64 {
  self.product_bytes
}
