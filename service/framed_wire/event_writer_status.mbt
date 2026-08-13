///|
/// Scalar direct-token writer for an already-authenticated owner-resident
/// decode. Failure leaves writer credit and storage unchanged.
pub fn CanonicalEventWriter::prepare_token_bytes_status(
  self : CanonicalEventWriter,
  token_id~ : Int,
  position~ : Int,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Bool {
  let inference = self.limits.inference_limits()
  if self.busy ||
    token_id < 0 ||
    token_id > inference.max_token_id() ||
    position < 0 ||
    position >= inference.max_new_tokens() ||
    source_offset < 0 ||
    length < 0 ||
    source_offset > source.length() ||
    length > source.length() - source_offset ||
    length > inference.max_decoded_delta_bytes() ||
    length > self.limits.max_frame_bytes() - EVENT_HEADER_BYTES ||
    !valid_utf8_range(source, source_offset, length) {
    return false
  }
  let total = EVENT_HEADER_BYTES + length
  prepare_event_header(
    self.scratch,
    2,
    total,
    if length == 0 {
      0
    } else {
      1
    },
    length,
  )
  for index in 0..<length {
    self.scratch[EVENT_HEADER_BYTES + index] = source[source_offset + index]
  }
  write_i32(self.scratch, 24, token_id)
  write_i32(self.scratch, 28, position)
  write_u32(self.scratch, 16, frame_checksum(self.scratch, total))
  self.publish(total)
  true
}
