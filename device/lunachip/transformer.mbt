///|
pub(all) enum ReductionOperation {
  Sum
  SumSquares
  Maximum
}

///|
pub(all) enum UnaryOperation {
  Reciprocal
  ReciprocalSqrt
  Exp
  NegativeAbs
}

///|
pub(all) enum ScalarOperand {
  Broadcast(Int32)
  Immediate(Float)
}

///|
fn no_partial_alias(
  dst : Int32,
  source : Int32,
  count : Int32,
) -> Unit raise AdapterError {
  if dst != source && dst < source + count && source < dst + count {
    raise InvalidInstruction
  }
}

///|
fn fields(
  op : UInt32,
  dst : Int32,
  a : Int32,
  b : Int32,
  n : Int32,
  k~ : Int32 = 0,
  bs~ : Int32 = 1,
  br~ : Int32 = 0,
  immediate~ : UInt32 = 0U,
  flags~ : UInt32 = 0U,
) -> Instruction {
  {
    words: ReadOnlyArray::from_array([
      op,
      dst.reinterpret_as_uint(),
      a.reinterpret_as_uint(),
      b.reinterpret_as_uint(),
      n.reinterpret_as_uint(),
      k.reinterpret_as_uint(),
      bs.reinterpret_as_uint(),
      br.reinterpret_as_uint(),
      0U,
      immediate,
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
pub fn Instruction::scalar(
  operation : BinaryOperation,
  dst : Int32,
  source : Int32,
  operand : ScalarOperand,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  scratch(source, count)
  no_partial_alias(dst, source, count)
  let (b, immediate, flags) = match operand {
    Broadcast(b) => {
      scratch(b, 1)
      // Each element re-reads the broadcast; writing it would alter later elements.
      if count > 1 && b >= dst && b < dst + count {
        raise InvalidInstruction
      }
      (b, 0U, 1U)
    }
    Immediate(value) => (0, value.reinterpret_as_uint(), 2U)
  }
  fields(
    match operation {
      Add => 2U
      Multiply => 3U
    },
    dst,
    source,
    b,
    count,
    immediate~,
    flags~,
  )
}

///|
pub fn Instruction::reduce(
  operation : ReductionOperation,
  dst : Int32,
  source : Int32,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, 1)
  scratch(source, count)
  fields(
    match operation {
      Sum => 6U
      SumSquares => 7U
      Maximum => 8U
    },
    dst,
    source,
    0,
    count,
  )
}

///|
/// Uses the published approximate scalar functions, not native IEEE libm.
pub fn Instruction::unary(
  operation : UnaryOperation,
  dst : Int32,
  source : Int32,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  scratch(source, count)
  no_partial_alias(dst, source, count)
  fields(
    match operation {
      Reciprocal => 9U
      ReciprocalSqrt => 10U
      Exp => 11U
      NegativeAbs => 12U
    },
    dst,
    source,
    0,
    count,
  )
}

///|
/// Stable SiLU numerator x * (x < 0 ? exp(-abs(x)) : 1).
pub fn Instruction::silu_numerator(
  dst : Int32,
  input : Int32,
  exponential : Int32,
  count : Int32,
) -> Self raise AdapterError {
  scratch(dst, count)
  scratch(input, count)
  scratch(exponential, count)
  no_partial_alias(dst, input, count)
  no_partial_alias(dst, exponential, count)
  fields(13U, dst, input, exponential, count)
}

///|
/// Ordered, separate FP32 multiply and add from +0. Strides are in words.
pub fn Instruction::scratch_gemv(
  dst : Int32,
  input : Int32,
  matrix : Int32,
  rows : Int32,
  columns : Int32,
  column_stride : Int32,
  row_stride : Int32,
) -> Self raise AdapterError {
  scratch(dst, rows)
  scratch(input, columns)
  if matrix < 0 ||
    column_stride < 1 ||
    column_stride > 32767 ||
    row_stride < 0 ||
    row_stride > 32767 {
    raise InvalidInstruction
  }
  let end = matrix.to_int64() +
    (rows - 1).to_int64() * row_stride.to_int64() +
    (columns - 1).to_int64() * column_stride.to_int64() +
    1L
  if end > 32768L ||
    (dst < input + columns && input < dst + rows) ||
    (dst.to_int64() < end && matrix < dst + rows) {
    raise InvalidInstruction
  }
  fields(
    5U,
    dst,
    input,
    matrix,
    rows,
    k=columns,
    bs=column_stride,
    br=row_stride,
  )
}
