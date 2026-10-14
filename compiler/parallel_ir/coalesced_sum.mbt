///|
/// Concatenation of independent, equally typed reductions. This is not an
/// algebraic reassociation: lanes never add to one another and consumers retain
/// their original publication/rounding. Both producers must finish before pack.
pub struct CoalescedSumPlan {
  priv rows : Int
  priv left : Int
  priv right : Int
  priv bytes : Int64
}

///|
pub fn CoalescedSumPlan::new(
  maximum_rows~ : Int,
  left_width~ : Int,
  right_width~ : Int,
) -> Self raise ParallelError {
  guard maximum_rows > 0 &&
    left_width > 0 &&
    right_width > 0 &&
    left_width.to_int64() + right_width.to_int64() <= 2147483647L &&
    maximum_rows.to_int64() * (left_width.to_int64() + right_width.to_int64()) <=
    2147483647L * 256L else {
    raise Capacity
  }
  {
    rows: maximum_rows,
    left: left_width,
    right: right_width,
    bytes: maximum_rows.to_int64() *
    (left_width.to_int64() + right_width.to_int64()) *
    4L,
  }
}

///|
pub fn CoalescedSumPlan::geometry(self : Self) -> (Int, Int, Int) {
  (self.rows, self.left, self.right)
}

///|
pub fn CoalescedSumPlan::width(self : Self) -> Int {
  self.left + self.right
}

///|
pub fn CoalescedSumPlan::storage_bytes(self : Self) -> Int64 {
  self.bytes
}
