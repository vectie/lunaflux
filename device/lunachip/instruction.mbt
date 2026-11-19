///|
/// A validated instruction in the existing 16-word Qwen layer ISA.
pub struct Instruction {
  priv words : ReadOnlyArray[UInt32]
}

///|
fn scratch(base : Int32, count : Int32) -> Unit raise AdapterError {
  if base < 0 || count < 1 || count > 4096 || base > 32768 - count {
    raise InvalidInstruction
  }
}

///|
fn instruction(
  op : UInt32,
  dst : Int32,
  a : Int32,
  b : Int32,
  n : Int32,
  ddr : UInt32,
  flags~ : UInt32 = 0U,
) -> Instruction {
  {
    words: ReadOnlyArray::from_array([
      op,
      dst.reinterpret_as_uint(),
      a.reinterpret_as_uint(),
      b.reinterpret_as_uint(),
      n.reinterpret_as_uint(),
      0U,
      1U,
      0U,
      ddr,
      0U,
      flags,
      0U,
      0U,
      0U,
      0U,
      0U,
    ]),
  }
}

///|
pub fn Instruction::halt() -> Self {
  instruction(0U, 0, 0, 0, 0, 0U)
}

///|
pub fn Instruction::copy(
  dst : Int32,
  source : Int32,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  scratch(source, count)
  // Exact aliases are safe; partial overlap has sequential copy semantics and
  // is not admitted as an implementation of a parallel tensor operation.
  if dst != source && dst < source + count && source < dst + count {
    raise InvalidInstruction
  }
  instruction(1U, dst, source, 0, count, 0U)
}

///|
pub(all) enum BinaryOperation {
  Add
  Multiply
}

///|
pub fn Instruction::binary(
  operation : BinaryOperation,
  dst : Int32,
  left : Int32,
  right : Int32,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  scratch(left, count)
  scratch(right, count)
  for source in [left, right] {
    if dst != source && dst < source + count && source < dst + count {
      raise InvalidInstruction
    }
  }
  instruction(
    match operation {
      Add => 2U
      Multiply => 3U
    },
    dst,
    left,
    right,
    count,
    0U,
  )
}

///|
pub fn Instruction::load(
  device : DeviceContract,
  dst : Int32,
  address : UInt64,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  device.check_range(address, count.to_uint64() * 4UL)
  if address % 4UL != 0UL {
    raise InvalidRange
  }
  instruction(14U, dst, 0, 0, count, address.to_uint())
}

///|
/// Read the little-endian DDR address at pointer, then expand complete lines
/// of BF16 into FP32 scratch. Requires the PBlaze3 BF16 capability register.
pub fn Instruction::load_indirect_bf16(
  device : DeviceContract,
  dst : Int32,
  pointer : UInt64,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  device.check_range(pointer, 4UL)
  if pointer % 4UL != 0UL || count % 64 != 0 {
    raise InvalidRange
  }
  instruction(14U, dst, 0, 0, count, pointer.to_uint(), flags=1U)
}

///|
pub fn Instruction::store(
  device : DeviceContract,
  source : Int32,
  address : UInt64,
  count : Int32,
) -> Self raise AdapterError {
  scratch(source, count)
  device.check_range(address, count.to_uint64() * 4UL)
  if address % 4UL != 0UL {
    raise InvalidRange
  }
  instruction(15U, 0, source, 0, count, address.to_uint())
}

///|
pub fn Instruction::words(self : Self) -> ReadOnlyArray[UInt32] {
  self.words
}
