///|
const MAX_STEP_ROWS : Int = 4096

///|
const MAX_STEP_CELLS : Int = 1048576

///|
const INT32_MAX : Int = 2147483647

///|
const INT32_BYTES : Int = 4

///|
const COUNTS_LENGTH : Int = 5

///|
const DEVICE_BUFFER_ALIGNMENT : Int64 = 16L

///|
pub fn DeviceStepLimits::new(
  max_rows~ : Int,
  max_tokens~ : Int,
  max_pages~ : Int,
  max_sequence_tokens~ : Int,
  max_token_id~ : Int,
) -> DeviceStepLimits raise DeviceStepError {
  if max_rows <= 0 || max_rows > MAX_STEP_ROWS {
    raise InvalidLimit(Rows)
  }
  if max_tokens <= 0 || max_tokens > MAX_STEP_CELLS {
    raise InvalidLimit(Tokens)
  }
  if max_pages <= 0 || max_pages > MAX_STEP_CELLS {
    raise InvalidLimit(Pages)
  }
  if max_sequence_tokens <= 0 || max_sequence_tokens > INT32_MAX {
    raise InvalidLimit(SequenceTokens)
  }
  if max_token_id < 0 || max_token_id > INT32_MAX {
    raise InvalidLimit(TokenId)
  }
  if max_rows > INT32_MAX / INT32_BYTES - 1 ||
    max_tokens > INT32_MAX / INT32_BYTES ||
    max_pages > INT32_MAX / INT32_BYTES {
    raise InvalidLimit(ByteCapacity)
  }
  { max_rows, max_tokens, max_pages, max_sequence_tokens, max_token_id }
}

///|
fn row_offset_cells(limits : DeviceStepLimits) -> Int {
  limits.max_rows() + 1
}

///|
fn byte_capacity(cells : Int) -> Int64 {
  Int64::from_int(cells * INT32_BYTES)
}
