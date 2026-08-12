///|
fn byte_from_bytelevel_scalar(scalar : Int) -> Int? {
  if (scalar >= 0x21 && scalar <= 0x7e) ||
    (scalar >= 0xa1 && scalar <= 0xac) ||
    (scalar >= 0xae && scalar <= 0xff) {
    Some(scalar)
  } else if scalar >= 0x100 && scalar <= 0x120 {
    Some(scalar - 0x100)
  } else if scalar >= 0x121 && scalar <= 0x142 {
    Some(0x7f + scalar - 0x121)
  } else if scalar == 0x143 {
    Some(0xad)
  } else {
    None
  }
}

///|
/// Reverse the canonical GPT-2/ByteLevel byte-to-Unicode alphabet.
fn decode_bytelevel_piece(
  piece : String,
  entry_index : Int,
) -> Bytes raise TokenizerJsonError {
  let output : Array[Byte] = []
  let mut character_index = 0
  for character in piece {
    match byte_from_bytelevel_scalar(character.to_int()) {
      Some(value) => output.push(value.to_byte())
      None => raise InvalidByteLevelPiece(entry_index~, character_index~)
    }
    character_index += 1
  }
  Bytes::from_array(output)
}

///|
fn join_bytes(left : Bytes, right : Bytes) -> Bytes {
  let output : Array[Byte] = Array::new(capacity=left.length() + right.length())
  output.push_iter(left.iter())
  output.push_iter(right.iter())
  Bytes::from_array(output)
}
