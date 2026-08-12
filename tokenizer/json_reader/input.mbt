///|
const MAX_SUPPORTED_LIMIT : Int = 1_073_741_824

///|
fn validate_reader_limits(
  max_bytes : Int,
  max_depth : Int,
) -> Unit raise TokenizerJsonError {
  if max_bytes <= 0 || max_bytes > MAX_SUPPORTED_LIMIT {
    raise InvalidByteLimit
  }
  if max_depth <= 0 || max_depth > 256 {
    raise InvalidDepthLimit
  }
}

///|
fn validate_utf8_size(
  input : StringView,
  max_bytes : Int,
) -> Unit raise TokenizerJsonError {
  let mut index = 0
  let mut byte_count = 0
  while index < input.length() {
    let unit = input[index]
    let (width, next_index) = if unit <= 0x7f {
      (1, index + 1)
    } else if unit <= 0x7ff {
      (2, index + 1)
    } else if unit.is_leading_surrogate() {
      if index + 1 >= input.length() ||
        !input[index + 1].is_trailing_surrogate() {
        raise InvalidUtf8
      }
      (4, index + 2)
    } else if unit.is_trailing_surrogate() {
      raise InvalidUtf8
    } else {
      (3, index + 1)
    }
    if byte_count > max_bytes - width {
      raise ByteLimitExceeded(limit=max_bytes)
    }
    byte_count += width
    index = next_index
  }
}
