///|
fn Qwen3TokenSseBuffer::append(
  self : Qwen3TokenSseBuffer,
  offset : Int,
  bytes : BytesView,
) -> Int {
  for index, byte in bytes {
    self.storage[offset + index] = byte
  }
  offset + bytes.length()
}

///|
pub fn Qwen3TokenSseBuffer::stage(
  self : Qwen3TokenSseBuffer,
  token_id : Int,
) -> Bytes {
  let mut offset = self.append(
    0, b"data: {\"schema\":\"lunaflux.benchmark-token.v1\",\"token_id\":",
  )
  let mut remaining = token_id
  let mut digits = 0
  if remaining == 0 {
    self.digits[0] = b'0'
    digits = 1
  } else {
    while remaining > 0 {
      self.digits[digits] = b'0' + (remaining % 10).to_byte()
      remaining /= 10
      digits += 1
    }
  }
  for index in 0..<digits {
    self.storage[offset + index] = self.digits[digits - index - 1]
  }
  offset += digits
  offset = self.append(offset, b",\"text\":\"\"}\n\n")
  Bytes::makei(offset, index => self.storage[index])
}

///|
pub fn qwen3_terminal_sse_event() -> Bytes {
  b"data: {\"schema\":\"lunaflux.benchmark-terminal.v1\"}\n\ndata: [DONE]\n\n"
}
