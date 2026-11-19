///|
/// Pure BF16 row concatenation. This plan introduces no transpose, arithmetic
/// or storage round: separate packed Q/K/V rows become the existing QKV ABI.
pub struct PackedQkvPrecision {
  priv query_width : Int32
  priv kv_width : Int32
  priv rows : Int32
}

///|
pub fn PackedQkvPrecision::new(
  query_width~ : Int32,
  kv_width~ : Int32,
  maximum_rows~ : Int32,
) -> Self raise PrecisionPlanError {
  guard query_width > 0 &&
    kv_width > 0 &&
    maximum_rows > 0 &&
    query_width.to_int64() + kv_width.to_int64() * 2L <= 2147483647L else {
    raise IllegalPlacement
  }
  ignore(
    expert_checked_mul(
      maximum_rows.to_int64(),
      (query_width.to_int64() + kv_width.to_int64() * 2L) * 2L,
    ),
  )
  { query_width, kv_width, rows: maximum_rows, }
}

///|
pub fn PackedQkvPrecision::geometry(self : Self) -> (Int32, Int32, Int32) {
  (self.query_width, self.kv_width, self.rows)
}

///|
pub fn PackedQkvPrecision::output_bytes(self : Self) -> Int64 {
  self.rows.to_int64() *
  (self.query_width.to_int64() + self.kv_width.to_int64() * 2L) *
  2L
}
