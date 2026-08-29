///|
fn checked_nonnegative_add(
  left : Int64,
  right : Int64,
  area : TensorParallelAdmissionArea,
  rank : Int,
) -> Int64 raise TensorParallelAdmissionError {
  if left < 0L || right < 0L {
    raise reject(area, InvalidMemory, rank)
  }
  if left > MAX_INT64 - right {
    raise reject(area, Overflow, rank)
  }
  left + right
}

///|
fn reservation_total(
  weight : Int64,
  activation_workspace : Int64,
  kv : Int64,
  collective : Int64,
  rank : Int,
) -> Int64 raise TensorParallelAdmissionError {
  checked_nonnegative_add(
    checked_nonnegative_add(
      checked_nonnegative_add(weight, activation_workspace, RankMemory, rank),
      kv,
      RankMemory,
      rank,
    ),
    collective,
    RankMemory,
    rank,
  )
}
