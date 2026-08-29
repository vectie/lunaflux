///|
fn inference_credential_header_length(source : FixedArray[Byte]) -> Int {
  if source.length() != INHERITED_INFERENCE_CREDENTIAL_HEADER_BYTES ||
    source[0] != b'L' ||
    source[1] != b'F' ||
    source[2] != b'C' ||
    source[3] != b'1' ||
    source[4] != b'K' ||
    source[5] != b'E' ||
    source[6] != b'Y' ||
    source[7] != b'\n' ||
    source[10] != b'\x00' ||
    source[11] != b'\x00' {
    return 0
  }
  let length = source[8].to_int() | (source[9].to_int() << 8)
  if length <= 0 || length > INHERITED_INFERENCE_CREDENTIAL_MAX_BYTES {
    return 0
  }
  length
}

///|
fn wipe_credential_bytes(
  header : FixedArray[Byte],
  credential : FixedArray[Byte],
) -> Unit {
  for index in 0..<header.length() {
    header[index] = b'\x00'
  }
  for index in 0..<credential.length() {
    credential[index] = b'\x00'
  }
}
