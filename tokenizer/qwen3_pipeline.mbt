///|
fn qwen3_scalar_at(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
) -> (Int, Int) raise TokenizerError {
  if offset < 0 || offset >= length {
    raise InvalidUtf8Input(byte_offset=Int::max(offset, 0))
  }
  let first = input[offset].to_int()
  if first <= 0x7f {
    return (first, 1)
  }
  let width = if first >= 0xc2 && first <= 0xdf {
    2
  } else if first >= 0xe0 && first <= 0xef {
    3
  } else if first >= 0xf0 && first <= 0xf4 {
    4
  } else {
    raise InvalidUtf8Input(byte_offset=offset)
  }
  if offset + width > length {
    raise InvalidUtf8Input(byte_offset=offset)
  }
  for index in 1..<width {
    let continuation = input[offset + index].to_int()
    if continuation < 0x80 || continuation > 0xbf {
      raise InvalidUtf8Input(byte_offset=offset)
    }
  }
  let second = input[offset + 1].to_int()
  if (first == 0xe0 && second < 0xa0) ||
    (first == 0xed && second > 0x9f) ||
    (first == 0xf0 && second < 0x90) ||
    (first == 0xf4 && second > 0x8f) {
    raise InvalidUtf8Input(byte_offset=offset)
  }
  let scalar = match width {
    2 => ((first & 0x1f) << 6) | (second & 0x3f)
    3 =>
      ((first & 0x0f) << 12) |
      ((second & 0x3f) << 6) |
      (input[offset + 2].to_int() & 0x3f)
    _ =>
      ((first & 0x07) << 18) |
      ((second & 0x3f) << 12) |
      ((input[offset + 2].to_int() & 0x3f) << 6) |
      (input[offset + 3].to_int() & 0x3f)
  }
  (scalar, width)
}

///|

///|
fn qwen3_is_number(scalar : Int) -> Bool {
  scalar.to_char().unwrap().is_numeric()
}

///|
fn qwen3_is_whitespace(scalar : Int) -> Bool {
  scalar.to_char().unwrap().is_whitespace()
}

///|
fn qwen3_ascii_lower(value : Byte) -> Byte {
  if value >= b'A' && value <= b'Z' {
    (value.to_int() + 0x20).to_byte()
  } else {
    value
  }
}

///|
fn qwen3_matches_ascii_casefold(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
  expected : Bytes,
) -> Bool {
  if offset > length - expected.length() {
    return false
  }
  for index, byte in expected {
    if qwen3_ascii_lower(input[offset + index]) != byte {
      return false
    }
  }
  true
}

///|
fn qwen3_contraction_end(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
) -> Int {
  if offset >= length || input[offset] != b'\'' {
    return -1
  }
  if qwen3_matches_ascii_casefold(input, length, offset, b"'s") ||
    qwen3_matches_ascii_casefold(input, length, offset, b"'t") ||
    qwen3_matches_ascii_casefold(input, length, offset, b"'m") ||
    qwen3_matches_ascii_casefold(input, length, offset, b"'d") {
    offset + 2
  } else if qwen3_matches_ascii_casefold(input, length, offset, b"'re") ||
    qwen3_matches_ascii_casefold(input, length, offset, b"'ve") ||
    qwen3_matches_ascii_casefold(input, length, offset, b"'ll") {
    offset + 3
  } else {
    -1
  }
}

///|
fn qwen3_scan_letters(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
  include_marks~ : Bool,
) -> Int raise TokenizerError {
  let mut cursor = offset
  while cursor < length {
    let (scalar, width) = qwen3_scalar_at(input, length, cursor)
    if !qwen3_is_letter(scalar) && !(include_marks && qwen35_is_mark(scalar)) {
      break
    }
    cursor += width
  }
  cursor
}

///|
fn qwen3_pretoken_end(
  input : FixedArray[Byte],
  length : Int,
  offset : Int,
  include_marks~ : Bool,
) -> Int raise TokenizerError {
  let contraction = qwen3_contraction_end(input, length, offset)
  if contraction >= 0 {
    return contraction
  }
  let (first, first_width) = qwen3_scalar_at(input, length, offset)
  if qwen3_is_letter(first) || (include_marks && qwen35_is_mark(first)) {
    return qwen3_scan_letters(input, length, offset, include_marks~)
  }
  if first != 0x0d &&
    first != 0x0a &&
    !qwen3_is_letter(first) &&
    !qwen3_is_number(first) &&
    offset + first_width < length {
    let next_offset = offset + first_width
    let (next, _) = qwen3_scalar_at(input, length, next_offset)
    if qwen3_is_letter(next) || (include_marks && qwen35_is_mark(next)) {
      return qwen3_scan_letters(input, length, next_offset, include_marks~)
    }
  }
  if qwen3_is_number(first) {
    return offset + first_width
  }
  let mut symbol_start = offset
  if first == 0x20 && offset + 1 < length {
    let (next, _) = qwen3_scalar_at(input, length, offset + 1)
    if !qwen3_is_whitespace(next) &&
      !qwen3_is_letter(next) &&
      !(include_marks && qwen35_is_mark(next)) &&
      !qwen3_is_number(next) {
      symbol_start = offset + 1
    }
  }
  let (symbol, _) = qwen3_scalar_at(input, length, symbol_start)
  if !qwen3_is_whitespace(symbol) &&
    !qwen3_is_letter(symbol) &&
    !(include_marks && qwen35_is_mark(symbol)) &&
    !qwen3_is_number(symbol) {
    let mut cursor = symbol_start
    while cursor < length {
      let (scalar, width) = qwen3_scalar_at(input, length, cursor)
      if qwen3_is_whitespace(scalar) ||
        qwen3_is_letter(scalar) ||
        (include_marks && qwen35_is_mark(scalar)) ||
        qwen3_is_number(scalar) {
        break
      }
      cursor += width
    }
    while cursor < length && (input[cursor] == b'\r' || input[cursor] == b'\n') {
      cursor += 1
    }
    return cursor
  }
  if qwen3_is_whitespace(first) {
    let mut cursor = offset
    let mut newline_end = -1
    while cursor < length {
      let (scalar, width) = qwen3_scalar_at(input, length, cursor)
      if !qwen3_is_whitespace(scalar) {
        break
      }
      cursor += width
      if scalar == 0x0d || scalar == 0x0a {
        newline_end = cursor
      }
    }
    if newline_end >= 0 {
      return newline_end
    }
    if cursor < length && cursor - offset > first_width {
      return cursor - first_width
    }
    return cursor
  }
  offset + first_width
}
