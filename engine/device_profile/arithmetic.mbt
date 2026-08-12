///|
const MAX_INT64 : Int64 = 9223372036854775807L

///|
const MAX_PROFILE_REGION_ID : Int64 = 2147483647L

///|
const BF16_BYTES : Int64 = 2L

///|
const TOKEN_ID_BYTES : Int64 = 4L

///|
fn exact_shape(
  phase : ExecutionPhase,
  batch_rows : Int,
  query_tokens_per_row : Int,
  context_tokens : Int,
) -> ExactExecutionShape raise DeviceProfileError {
  if batch_rows <= 0 {
    raise InvalidShape(NonPositiveBatchRows)
  }
  if query_tokens_per_row <= 0 {
    raise InvalidShape(NonPositiveQueryTokens)
  }
  let batch = Int64::from_int(batch_rows)
  let query = Int64::from_int(query_tokens_per_row)
  if query != 0L && batch > MAX_INT64 / query {
    raise InvalidShape(TokenRowOverflow)
  }
  {
    phase,
    batch_rows,
    query_tokens_per_row,
    context_tokens,
    token_rows: batch * query,
  }
}

///|
fn checked_add(
  left : Int64,
  right : Int64,
  operation : @plan.OperationId,
) -> Int64 raise DeviceProfileError {
  if left < 0L || right < 0L || left > MAX_INT64 - right {
    raise SizeOverflow(operation)
  }
  left + right
}

///|
fn checked_multiply(
  left : Int64,
  right : Int64,
  operation : @plan.OperationId,
) -> Int64 raise DeviceProfileError {
  if left < 0L || right < 0L || (right != 0L && left > MAX_INT64 / right) {
    raise SizeOverflow(operation)
  }
  left * right
}

///|
fn checked_token_bytes(rows : Int64) -> Int64 raise DeviceProfileError {
  if rows < 0L || rows > MAX_INT64 / TOKEN_ID_BYTES {
    raise TokenSizeOverflow
  }
  rows * TOKEN_ID_BYTES
}

///|
fn checked_region_id(value : Int64) -> ProfileRegionId raise DeviceProfileError {
  if value < 0L || value > MAX_PROFILE_REGION_ID {
    raise InvalidRegion(ProfileRegionId(0), IdentifierOverflow)
  }
  ProfileRegionId(value.to_int())
}

///|
fn checked_region_id_add(
  left : Int64,
  right : Int64,
) -> ProfileRegionId raise DeviceProfileError {
  if left < 0L ||
    right < 0L ||
    right > MAX_PROFILE_REGION_ID ||
    left > MAX_PROFILE_REGION_ID - right {
    raise InvalidRegion(ProfileRegionId(0), IdentifierOverflow)
  }
  ProfileRegionId((left + right).to_int())
}

///|
fn is_power_of_two(value : Int64) -> Bool {
  value > 0L && (value & (value - 1L)) == 0L
}

///|
fn checked_region_end(
  view : ExecutionRegionView,
) -> Int64 raise DeviceProfileError {
  if view.offset_bytes < 0L {
    raise InvalidRegion(view.id, NegativeOffset)
  }
  if view.logical_bytes < 0L || view.reserved_bytes < 0L {
    raise InvalidRegion(view.id, NegativeLength)
  }
  if !is_power_of_two(view.alignment_bytes) {
    raise InvalidRegion(view.id, InvalidAlignment)
  }
  if view.offset_bytes % view.alignment_bytes != 0L {
    raise InvalidRegion(view.id, MisalignedOffset)
  }
  if view.logical_bytes > view.reserved_bytes {
    raise InvalidRegion(view.id, LogicalExceedsReserved)
  }
  if view.offset_bytes > MAX_INT64 - view.reserved_bytes {
    raise InvalidRegion(view.id, OutsideArena)
  }
  view.offset_bytes + view.reserved_bytes
}

///|
fn validate_region_capacity(
  view : ExecutionRegionView,
  arena_bytes : Int64,
) -> Unit raise DeviceProfileError {
  if arena_bytes < 0L || checked_region_end(view) > arena_bytes {
    raise InvalidRegion(view.id, OutsideArena)
  }
}
