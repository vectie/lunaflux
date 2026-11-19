///|
/// BF16 sigmoid and product with a strided raw projection gate view.
pub fn pointwise_gate_into(
  plan : @precision.PointwiseGatePrecision,
  input : ArrayView[Float],
  gate : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (width, vectors, maximum_rows) = plan.geometry()
  let (vector_stride, row_stride, offset) = plan.gate_layout()
  let packed = rows.to_int64() * vectors.to_int64() * width.to_int64()
  let required = (rows.to_int64() - 1L) * row_stride +
    (vectors - 1).to_int64() * vector_stride.to_int64() +
    offset.to_int64() +
    width.to_int64()
  guard rows > 0 &&
    rows <= maximum_rows &&
    input.length().to_int64() == packed &&
    output.length() == input.length() &&
    gate.length().to_int64() >= required else {
    fail("invalid gate buffers")
  }
  for row in 0..<rows {
    for vector in 0..<vectors {
      let gate_base = (row.to_int64() * row_stride).to_int() +
        vector * vector_stride +
        offset
      let base = (row * vectors + vector) * width
      for channel in 0..<width {
        output[base + channel] = round_bf16(
          round_bf16(input[base + channel]) *
          sigmoid_bf16(gate[gate_base + channel]),
        )
      }
    }
  }
}
