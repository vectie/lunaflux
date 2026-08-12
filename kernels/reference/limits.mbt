///|
/// Allocation ceiling for correctness-only reference evaluation.
///
/// Every matrix created by this package is checked against all three limits
/// before allocation. These limits are deliberately caller-owned; this package
/// has no global execution policy.
pub struct ReferenceLimits {
  max_rows : Int
  max_columns : Int
  max_elements : Int
} derive(Debug, Eq)

///|
pub fn ReferenceLimits::new(
  max_rows~ : Int,
  max_columns~ : Int,
  max_elements~ : Int,
) -> ReferenceLimits raise ReferenceKernelError {
  if max_rows <= 0 {
    raise InvalidLimit(MaxRows)
  }
  if max_columns <= 0 {
    raise InvalidLimit(MaxColumns)
  }
  if max_elements <= 0 {
    raise InvalidLimit(MaxElements)
  }
  { max_rows, max_columns, max_elements }
}

///|
pub fn ReferenceLimits::max_rows(self : ReferenceLimits) -> Int {
  self.max_rows
}

///|
pub fn ReferenceLimits::max_columns(self : ReferenceLimits) -> Int {
  self.max_columns
}

///|
pub fn ReferenceLimits::max_elements(self : ReferenceLimits) -> Int {
  self.max_elements
}

///|
fn checked_element_count(
  rows : Int,
  columns : Int,
  limits : ReferenceLimits,
) -> Int raise ReferenceKernelError {
  if rows <= 0 || columns <= 0 {
    raise InvalidShape(PositiveDimensions)
  }
  if rows > limits.max_rows {
    raise LimitExceeded(MaxRows)
  }
  if columns > limits.max_columns {
    raise LimitExceeded(MaxColumns)
  }
  let rows64 = Int64::from_int(rows)
  let columns64 = Int64::from_int(columns)
  if rows64 > 9223372036854775807L / columns64 {
    raise ArithmeticOverflow
  }
  let elements = rows64 * columns64
  if elements > Int64::from_int(limits.max_elements) {
    raise LimitExceeded(MaxElements)
  }
  if elements > 2147483647L {
    raise ArithmeticOverflow
  }
  elements.to_int()
}

///|
fn checked_sum(left : Int, right : Int) -> Int raise ReferenceKernelError {
  if left < 0 || right < 0 || left > 2147483647 - right {
    raise ArithmeticOverflow
  }
  left + right
}

///|
fn checked_product(left : Int, right : Int) -> Int raise ReferenceKernelError {
  if left < 0 || right < 0 || (right != 0 && left > 2147483647 / right) {
    raise ArithmeticOverflow
  }
  left * right
}
