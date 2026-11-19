///|
/// Row-major BF16 weights, FP32 activations, separate RNE FP32 products and
/// increasing-column FP32 accumulation from +0. No bias or fused multiply-add.
/// Scratch input/output must be disjoint, including exact aliases.
pub fn Instruction::bf16_gemv(
  device : DeviceContract,
  dst : Int32,
  input : Int32,
  weights : UInt64,
  rows : Int32,
  columns : Int32,
) -> Self raise AdapterError {
  scratch(dst, rows)
  scratch(input, columns)
  if columns % 64 != 0 || (dst < input + columns && input < dst + rows) {
    raise InvalidInstruction
  }
  device.check_range(weights, rows.to_uint64() * columns.to_uint64() * 2UL)
  if weights % 128UL != 0UL {
    raise InvalidRange
  }
  {
    words: ReadOnlyArray::from_array([
      4U,
      dst.reinterpret_as_uint(),
      input.reinterpret_as_uint(),
      0U,
      rows.reinterpret_as_uint(),
      columns.reinterpret_as_uint(),
      1U,
      0U,
      weights.to_uint(),
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
    ]),
  }
}
