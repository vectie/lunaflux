///|
/// Startup-prepared chunks. Each chunk ends with HALT and retains scratch
/// ownership until the entire program completes.
pub struct PreparedProgram {
  priv chunks : ReadOnlyArray[Bytes]
}

///|
pub fn PreparedProgram::new(
  operations : ArrayView[Instruction],
) -> Self raise AdapterError {
  if operations.is_empty() || operations.length() > 1048576 {
    raise InvalidProgram
  }
  for op in operations {
    if op.words[0] == 0U {
      raise InvalidProgram
    }
  }
  let chunks : Array[Bytes] = []
  let mut start = 0
  while start < operations.length() {
    let count = (operations.length() - start).min(2047)
    let bytes = Array::make((count + 1) * 64, b'\x00')
    for i in 0..<count {
      for j, word in operations[start + i].words {
        for k in 0..<4 {
          bytes[i * 64 + j * 4 + k] = (word >> (k * 8)).to_byte()
        }
      }
    }
    // The all-zero last instruction is the canonical HALT.
    chunks.push(Bytes::from_array(bytes[:]))
    start += count
  }
  { chunks: ReadOnlyArray::from_array(chunks), }
}

///|
pub fn PreparedProgram::chunk_count(self : Self) -> Int32 {
  self.chunks.length()
}

///|
pub fn PreparedProgram::chunk(self : Self, index : Int32) -> Bytes? {
  self.chunks.get(index)
}

///|
/// Consecutive programs share scratch and execute straight-line instructions.
/// Replace their boundary HALTs with one HALT per hardware instruction window.
pub fn PreparedProgram::concatenate(
  programs : ArrayView[PreparedProgram],
) -> Self raise AdapterError {
  if programs.is_empty() {
    raise InvalidProgram
  }
  let result : Array[Bytes] = []
  let buf = @buffer.Buffer(size_hint=131072)
  let halt = Bytes::make(64, b'\x00')
  for program in programs {
    for chunk in program.chunks {
      let end = chunk.length() - 64
      let mut offset = 0
      while offset < end {
        let n = (end - offset).min(2047 * 64 - buf.length())
        buf.write_bytes(chunk[offset:offset + n])
        offset += n
        if buf.length() == 2047 * 64 {
          buf.write_bytes(halt[:])
          result.push(buf.to_bytes())
          buf.reset()
        }
      }
    }
  }
  if !buf.is_empty() {
    buf.write_bytes(halt[:])
    result.push(buf.to_bytes())
  }
  { chunks: ReadOnlyArray::from_array(result), }
}
