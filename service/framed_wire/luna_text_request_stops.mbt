///|
pub fn LunaFramedTextRequestWrite::push_stop_token(
  self : Self,
  token_id : Int,
) -> Unit raise FramedWireError {
  let workspace = self.require()
  let inference = workspace.limits.inference_limits()
  if workspace.input_cursor != workspace.input_length ||
    workspace.stop_token_cursor >= workspace.stop_token_count ||
    workspace.stop_string_cursor != 0 ||
    workspace.stop_string_remaining != 0 ||
    token_id < 0 ||
    token_id > inference.max_token_id() {
    raise LunaFramedTextRequestInvalidRange
  }
  let offset = REQUEST_HEADER_BYTES +
    workspace.input_length +
    workspace.stop_token_cursor * 4
  write_i32(workspace.storage, offset, token_id)
  workspace.stop_token_cursor += 1
}

///|
pub fn LunaFramedTextRequestWrite::push_stop_string_length(
  self : Self,
  byte_length : Int,
) -> Unit raise FramedWireError {
  let workspace = self.require()
  let inference = workspace.limits.inference_limits()
  let remaining_strings = workspace.stop_string_count -
    workspace.stop_string_cursor -
    1
  let fixed_tail = workspace.cache_scope_length + remaining_strings * 4
  if workspace.input_cursor != workspace.input_length ||
    workspace.stop_token_cursor != workspace.stop_token_count ||
    workspace.stop_string_cursor >= workspace.stop_string_count ||
    workspace.stop_string_remaining != 0 ||
    byte_length <= 0 ||
    byte_length > inference.max_stop_string_bytes() ||
    remaining_strings < 0 ||
    workspace.payload_cursor > workspace.storage.length() ||
    fixed_tail < 0 ||
    fixed_tail > workspace.storage.length() - workspace.payload_cursor ||
    4 + byte_length >
    workspace.storage.length() - workspace.payload_cursor - fixed_tail {
    raise LunaFramedTextRequestInvalidRange
  }
  write_i32(workspace.storage, workspace.payload_cursor, byte_length)
  workspace.payload_cursor += 4
  workspace.stop_string_bytes += byte_length
  workspace.stop_string_remaining = byte_length
}

///|
pub fn LunaFramedTextRequestWrite::push_stop_string_byte(
  self : Self,
  byte : Byte,
) -> Unit raise FramedWireError {
  let workspace = self.require()
  if workspace.stop_string_remaining <= 0 ||
    workspace.payload_cursor >=
    workspace.storage.length() - workspace.cache_scope_length {
    raise LunaFramedTextRequestInvalidRange
  }
  workspace.storage[workspace.payload_cursor] = byte
  workspace.payload_cursor += 1
  workspace.stop_string_remaining -= 1
  if workspace.stop_string_remaining == 0 {
    workspace.stop_string_cursor += 1
  }
}
