///|
let plan_magic : UInt = 0x3158464c

///|
let wire_version : UInt = 1

///|
let plan_kind : UInt = 1

///|
let plan_header_bytes : Int = 64

///|
let plan_checksum_offset : Int = 60

///|
let prefill_row_bytes : Int = 96

///|
let decode_row_bytes : Int = 88

///|
let max_plan_frame_bytes : Int = 268435456

///|
fn checked_wire_add(left : Int, right : Int) -> Int raise WorkerWireError {
  if left < 0 || right < 0 || left > max_plan_frame_bytes - right {
    raise InvalidLimit(PlanFrameBytes)
  }
  left + right
}

///|
fn checked_wire_product(left : Int, right : Int) -> Int raise WorkerWireError {
  if left < 0 || right < 0 || (left > 0 && right > max_plan_frame_bytes / left) {
    raise InvalidLimit(PlanFrameBytes)
  }
  left * right
}

///|
pub fn plan_frame_byte_capacity(
  limits : @worker_protocol.WorkerProtocolLimits,
) -> Int raise WorkerWireError {
  let mut result = plan_header_bytes
  result = checked_wire_add(
    result,
    checked_wire_product(limits.max_plan_tokens(), 4),
  )
  result = checked_wire_add(
    result,
    checked_wire_product(limits.max_plan_pages(), 8),
  )
  result = checked_wire_add(
    result,
    checked_wire_product(limits.max_capabilities(), 4),
  )
  result = checked_wire_add(
    result,
    checked_wire_product(limits.max_prefill_rows(), prefill_row_bytes),
  )
  checked_wire_add(
    result,
    checked_wire_product(limits.max_decode_rows(), decode_row_bytes),
  )
}

///|
fn exact_plan_frame_bytes(
  token_count : Int,
  page_count : Int,
  capability_count : Int,
  prefill_count : Int,
  decode_count : Int,
) -> Int raise WorkerWireError {
  let mut result = plan_header_bytes
  result = checked_wire_add(result, checked_wire_product(token_count, 4))
  result = checked_wire_add(result, checked_wire_product(page_count, 8))
  result = checked_wire_add(result, checked_wire_product(capability_count, 4))
  result = checked_wire_add(
    result,
    checked_wire_product(prefill_count, prefill_row_bytes),
  )
  checked_wire_add(result, checked_wire_product(decode_count, decode_row_bytes))
}

///|
fn token_table_offset() -> Int {
  plan_header_bytes
}

///|
fn page_table_offset(token_count : Int) -> Int {
  plan_header_bytes + token_count * 4
}

///|
fn capability_table_offset(token_count : Int, page_count : Int) -> Int {
  page_table_offset(token_count) + page_count * 8
}

///|
fn prefill_table_offset(
  token_count : Int,
  page_count : Int,
  capability_count : Int,
) -> Int {
  capability_table_offset(token_count, page_count) + capability_count * 4
}

///|
fn decode_table_offset(
  token_count : Int,
  page_count : Int,
  capability_count : Int,
  prefill_count : Int,
) -> Int {
  prefill_table_offset(token_count, page_count, capability_count) +
  prefill_count * prefill_row_bytes
}
