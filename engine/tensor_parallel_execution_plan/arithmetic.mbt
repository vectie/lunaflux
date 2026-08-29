///|
const TP_EXEC_MAX_I64 : Int64 = 9223372036854775807L

///|
fn valid_alignment(value : Int64) -> Bool {
  value > 0L && value <= 65536L && (value & (value - 1L)) == 0L
}

///|
fn checked_add(
  left : Int64,
  right : Int64,
  operation? : Int = -1,
  operand? : Int = -1,
) -> Int64 raise TensorParallelExecutionPlanError {
  if left < 0L || right < 0L || left > TP_EXEC_MAX_I64 - right {
    raise rejected(CapacityOverflow, operation~, operand~)
  }
  left + right
}

///|
fn checked_multiply(
  left : Int64,
  right : Int64,
  operation? : Int = -1,
  operand? : Int = -1,
) -> Int64 raise TensorParallelExecutionPlanError {
  if left < 0L || right < 0L || (right != 0L && left > TP_EXEC_MAX_I64 / right) {
    raise rejected(CapacityOverflow, operation~, operand~)
  }
  left * right
}

///|
fn checked_align(
  value : Int64,
  alignment : Int64,
  operation? : Int = -1,
  operand? : Int = -1,
) -> Int64 raise TensorParallelExecutionPlanError {
  if !valid_alignment(alignment) {
    raise rejected(Alignment, operation~, operand~)
  }
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    checked_add(value, alignment - remainder, operation~, operand~)
  }
}

///|
fn scalar_region(
  space : TensorParallelMemorySpace,
  offset_bytes : Int64,
  byte_count : Int64,
  alignment : Int64,
  operation? : Int = -1,
  operand? : Int = -1,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  if offset_bytes < 0L || byte_count < 0L || !valid_alignment(alignment) {
    raise rejected(Alignment, operation~, operand~)
  }
  let reserved_bytes = checked_align(
    byte_count,
    alignment,
    operation~,
    operand~,
  )
  ignore(checked_add(offset_bytes, reserved_bytes, operation~, operand~))
  { space, offset_bytes, byte_count, reserved_bytes, alignment }
}
