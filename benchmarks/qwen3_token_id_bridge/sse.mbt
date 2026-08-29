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
fn Qwen3TokenSseBuffer::append_json_text(
  self : Qwen3TokenSseBuffer,
  offset : Int,
  text : BytesView,
) -> Int {
  let mut cursor = offset
  for byte in text {
    match byte {
      b'"' => cursor = self.append(cursor, b"\\\"")
      b'\\' => cursor = self.append(cursor, b"\\\\")
      b'\x08' => cursor = self.append(cursor, b"\\b")
      b'\x0c' => cursor = self.append(cursor, b"\\f")
      b'\n' => cursor = self.append(cursor, b"\\n")
      b'\r' => cursor = self.append(cursor, b"\\r")
      b'\t' => cursor = self.append(cursor, b"\\t")
      b'\x00'..=b'\x1f' => {
        let hex = b"0123456789abcdef"
        cursor = self.append(cursor, b"\\u00")
        self.storage[cursor] = hex[byte.to_int() >> 4]
        self.storage[cursor + 1] = hex[byte.to_int() & 0x0f]
        cursor += 2
      }
      _ => {
        self.storage[cursor] = byte
        cursor += 1
      }
    }
  }
  cursor
}

///|
pub fn Qwen3TokenSseBuffer::stage(
  self : Qwen3TokenSseBuffer,
  token_id : Int,
  text : BytesView,
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
  offset = self.append(offset, b",\"text\":\"")
  offset = self.append_json_text(offset, text)
  offset = self.append(offset, b"\"}\n\n")
  Bytes::makei(offset, index => self.storage[index])
}

///|
pub fn Qwen3TokenSseBuffer::terminal(
  self : Qwen3TokenSseBuffer,
  text : BytesView,
) -> Bytes {
  let mut offset = self.append(
    0, b"data: {\"schema\":\"lunaflux.benchmark-terminal.v1\",\"text\":\"",
  )
  offset = self.append_json_text(offset, text)
  offset = self.append(offset, b"\"}\n\ndata: [DONE]\n\n")
  Bytes::makei(offset, index => self.storage[index])
}
