///|
priv struct Source {
  file : @approved.ApprovedFile
  payload : Int64
  transfers : ReadOnlyArray[@qwen.WeightTransfer]
  layout : @layout.Layout
}

///|
fn read_range(
  file : @approved.ApprovedFile,
  offset : Int64,
  count : Int32,
) -> Bytes raise {
  let buffer = FixedArray::make(count, b'\x00')
  read_into(file, buffer, 0, offset, count)
  Bytes::from_array(buffer[:])
}

///|
fn read_into(
  file : @approved.ApprovedFile,
  buffer : FixedArray[Byte],
  destination : Int32,
  offset : Int64,
  count : Int32,
) -> Unit raise {
  let mut i = 0
  while i < count {
    let n = (count - i).min(1048576)
    file.read_exactly_at(
      buffer,
      destination_offset=destination + i,
      position=offset + i.to_int64(),
      byte_count=n,
    )
    i += n
  }
}

///|
fn Source::weight(
  self : Self,
  ordinal : Int32,
  offset~ : Int32 = 0,
  count~ : Int32 = -1,
) -> Bytes raise {
  let transfer = self.transfers[ordinal]
  let n = if count < 0 { transfer.source_bytes.to_int() } else { count }
  guard offset >= 0 &&
    n > 0 &&
    offset.to_int64() + n.to_int64() <= transfer.source_bytes else {
    fail("source tensor slice out of bounds")
  }
  read_range(
    self.file,
    self.payload + transfer.source_offset + offset.to_int64(),
    n,
  )
}

///|
fn Source::matrix(self : Self, address : UInt64, count : Int32) -> Bytes raise {
  guard count > 0 &&
    address < 0x80000000UL &&
    count.to_uint64() <= 0x80000000UL - address else {
    fail("matrix exceeds DDR aperture")
  }
  let bytes = FixedArray::make(count, b'\x00')
  let mut written = 0
  for t in self.transfers {
    let next = address + written.to_uint64()
    if next >= t.destination.address &&
      next - t.destination.address < t.destination.bytes {
      let offset = (next - t.destination.address).to_int64()
      let available = t.source_bytes - offset
      guard available > 0L && t.destination.bytes == t.source_bytes.to_uint64() else {
        fail("invalid matrix source extent")
      }
      let n = available.min((count - written).to_int64()).to_int()
      read_into(
        self.file,
        bytes,
        written,
        self.payload + t.source_offset + offset,
        n,
      )
      written += n
      if written == count {
        return Bytes::from_array(bytes[:])
      }
    }
  }
  fail("matrix spans unmapped DDR")
}

///|
fn bf16(bytes : Bytes, index : Int32) -> Float {
  Float::reinterpret_from_uint(
    (bytes[index * 2].to_uint() | (bytes[index * 2 + 1].to_uint() << 8)) << 16,
  )
}

///|
fn decode_word(bytes : Bytes, index : Int32) -> UInt32 {
  let i = index * 4
  bytes[i].to_uint() |
  (bytes[i + 1].to_uint() << 8) |
  (bytes[i + 2].to_uint() << 16) |
  (bytes[i + 3].to_uint() << 24)
}

///|
fn float_bytes(values : ArrayView[Float]) -> Bytes {
  let bytes = Array::make(values.length() * 4, b'\x00')
  for i, value in values {
    for k in 0..<4 {
      bytes[i * 4 + k] = (value.reinterpret_as_uint() >> (k * 8)).to_byte()
    }
  }
  Bytes::from_array(bytes[:])
}

///|
fn Source::embedding(self : Self, token : Int32) -> Array[Float] raise {
  guard token >= 0 && token < 151936 else { fail("invalid token") }
  let bytes = self.weight(0, offset=token * 2048, count=2048)
  Array::makei(1024, i => bf16(bytes, i))
}

///|
fn Source::step_input(
  self : Self,
  plan : @qwen.DecodeStep,
  token : Int32,
) -> Bytes raise {
  if plan.uses_indirect_bf16() {
    guard token >= 0 && token < 151936 && self.layout.weight(0) is Some(table) else {
      fail("invalid embedding row")
    }
    let offset = token.to_uint64() * 2048UL
    guard offset + 2048UL <= table.bytes else {
      fail("embedding row exceeds tied table")
    }
    words_bytes([(table.address + offset).to_uint()])
  } else {
    float_bytes(self.embedding(token)[:])
  }
}

///|
fn greedy(values : ArrayView[Float]) -> Int32 raise {
  guard values.length() == 151936 else { fail("incomplete vocabulary") }
  let mut best = 0
  for i, value in values {
    guard (value.reinterpret_as_uint() & 0x7f800000U) != 0x7f800000U else {
      fail("nonfinite logits")
    }
    if value > values[best] {
      best = i
    }
  }
  best
}

///|
/// EOS IDs from the admitted Qwen3-0.6B generation profile (im_end/endoftext).
fn terminal_token(token : Int32) -> Bool {
  token == 151645 || token == 151643
}
