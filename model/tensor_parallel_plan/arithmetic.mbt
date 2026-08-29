///|
const TP_PLAN_MAX_I64 : Int64 = 9223372036854775807L

///|
fn checked_add(
  left : Int64,
  right : Int64,
  index? : Int = -1,
) -> Int64 raise TensorParallelPlanError {
  if left < 0L || right < 0L || left > TP_PLAN_MAX_I64 - right {
    raise invalid(ArithmeticOverflow, index~)
  }
  left + right
}

///|
fn checked_multiply(
  left : Int64,
  right : Int64,
  index? : Int = -1,
) -> Int64 raise TensorParallelPlanError {
  if left < 0L || right < 0L || (right != 0L && left > TP_PLAN_MAX_I64 / right) {
    raise invalid(ArithmeticOverflow, index~)
  }
  left * right
}

///|
fn valid_alignment(value : Int64) -> Bool {
  value >= 16L && value <= 4096L && (value & (value - 1L)) == 0L
}
