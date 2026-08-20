///|
fn luna_event_invalid(rule : LunaEventRule) -> LunaEventError {
  LunaEventInvalid(rule)
}

///|
fn luna_valid_utf8_range(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Bool {
  let end = offset + length
  let mut index = offset
  while index < end {
    let first = source[index].to_int()
    if first < 0x80 {
      index += 1
    } else if first >= 0xc2 && first <= 0xdf {
      if index + 1 >= end {
        return false
      }
      let second = source[index + 1].to_int()
      if second < 0x80 || second > 0xbf {
        return false
      }
      index += 2
    } else if first >= 0xe0 && first <= 0xef {
      if index + 2 >= end {
        return false
      }
      let second = source[index + 1].to_int()
      let third = source[index + 2].to_int()
      if third < 0x80 ||
        third > 0xbf ||
        second < 0x80 ||
        second > 0xbf ||
        (first == 0xe0 && second < 0xa0) ||
        (first == 0xed && second > 0x9f) {
        return false
      }
      index += 3
    } else if first >= 0xf0 && first <= 0xf4 {
      if index + 3 >= end {
        return false
      }
      let second = source[index + 1].to_int()
      let third = source[index + 2].to_int()
      let fourth = source[index + 3].to_int()
      if third < 0x80 ||
        third > 0xbf ||
        fourth < 0x80 ||
        fourth > 0xbf ||
        second < 0x80 ||
        second > 0xbf ||
        (first == 0xf0 && second < 0x90) ||
        (first == 0xf4 && second > 0x8f) {
        return false
      }
      index += 4
    } else {
      return false
    }
  }
  true
}

///|
fn luna_valid_public_code_range(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Bool {
  if length <= 0 ||
    length > 64 ||
    offset < 0 ||
    offset > source.length() - length {
    return false
  }
  for index in offset..<(offset + length) {
    let byte = source[index]
    if !((byte >= b'a' && byte <= b'z') ||
      (byte >= b'A' && byte <= b'Z') ||
      (byte >= b'0' && byte <= b'9') ||
      byte == b'-' ||
      byte == b'_' ||
      byte == b'.' ||
      byte == b':') {
      return false
    }
  }
  true
}

///|
fn luna_valid_usage(
  limits : @inference.InferenceLimits,
  input_tokens : Int,
  cached_input_tokens : Int,
  output_tokens : Int,
) -> Bool {
  input_tokens >= 0 &&
  input_tokens <= limits.max_context_tokens() &&
  cached_input_tokens >= 0 &&
  cached_input_tokens <= input_tokens &&
  output_tokens >= 0 &&
  output_tokens <= limits.max_new_tokens() &&
  input_tokens <= limits.max_context_tokens() - output_tokens
}

///|
fn luna_valid_source_range(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
  maximum : Int,
) -> Bool {
  offset >= 0 &&
  length >= 0 &&
  offset <= source.length() &&
  length <= source.length() - offset &&
  length <= maximum
}
