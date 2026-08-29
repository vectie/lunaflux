///|
fn luna_openai_range_is_valid(
  source_length : Int,
  source_offset : Int,
  length : Int,
) -> Bool {
  source_offset >= 0 &&
  length >= 0 &&
  source_offset <= source_length &&
  length <= source_length - source_offset
}

///|
fn luna_openai_valid_utf8(
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
      if second < 0x80 ||
        second > 0xbf ||
        third < 0x80 ||
        third > 0xbf ||
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
      if second < 0x80 ||
        second > 0xbf ||
        third < 0x80 ||
        third > 0xbf ||
        fourth < 0x80 ||
        fourth > 0xbf ||
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
fn luna_openai_safe_id_byte(byte : Byte) -> Bool {
  (byte >= b'a' && byte <= b'z') ||
  (byte >= b'A' && byte <= b'Z') ||
  (byte >= b'0' && byte <= b'9') ||
  byte == b'-' ||
  byte == b'_'
}
