///|
fn checked_product(
  values : ArrayView[Int64],
) -> Int64 raise JointDiffusionReferenceError {
  let max = 9223372036854775807L
  let mut product = 1L
  for value in values {
    if value <= 0L || product > max / value {
      raise ShapeProductOverflow
    }
    product = product * value
  }
  product
}

///|
fn exact_int(value : Int64) -> Int raise JointDiffusionReferenceError {
  if value < 0L || value > 2147483647L {
    raise ShapeProductOverflow
  }
  value.to_int()
}
