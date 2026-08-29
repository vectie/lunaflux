///|
const LLAMA_TP_MAX_INT64 : Int64 = 9223372036854775807L

///|
fn llama_tp_checked_add(
  left : Int64,
  right : Int64,
  reference : Int,
) -> Int64 raise LlamaTensorParallelError {
  if left < 0L || right < 0L || left > LLAMA_TP_MAX_INT64 - right {
    raise ArithmeticOverflow(reference~)
  }
  left + right
}

///|
fn llama_tp_checked_mul(
  left : Int64,
  right : Int64,
  reference : Int,
) -> Int64 raise LlamaTensorParallelError {
  if left < 0L ||
    right < 0L ||
    (right != 0L && left > LLAMA_TP_MAX_INT64 / right) {
    raise ArithmeticOverflow(reference~)
  }
  left * right
}

///|
fn llama_tp_checked_align(
  value : Int64,
  alignment : Int64,
  reference : Int,
) -> Int64 raise LlamaTensorParallelError {
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    llama_tp_checked_add(value, alignment - remainder, reference)
  }
}

///|
fn llama_tp_valid_alignment(value : Int64) -> Bool {
  value >= 16L && value <= 4096L && (value & (value - 1L)) == 0L
}

///|
fn llama_tp_require_divisible(
  value : Int,
  world_size : Int,
  dimension : LlamaTensorParallelDimension,
) -> Unit raise LlamaTensorParallelError {
  if value <= 0 || value % world_size != 0 {
    raise NonDivisible(dimension)
  }
}
