///|
/// Offline ordered FP32 reference with explicit BF16 inputs and stage rounds.
/// The host sqrt/exp implementations are not a GPU bitwise oracle.
pub fn rms_norm_into(
  plan : @precision.RmsNormPrecision,
  input : ArrayView[Float],
  weight : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
  gate~ : ArrayView[Float] = [],
) -> Unit raise {
  let (width, vectors, maximum_rows) = plan.geometry()
  let (vector_stride, row_stride) = plan.input_layout()
  let required = (rows.to_int64() - 1L) * row_stride +
    (vectors - 1).to_int64() * vector_stride.to_int64() +
    width.to_int64()
  let packed = rows.to_int64() * vectors.to_int64() * width.to_int64()
  guard rows > 0 &&
    rows <= maximum_rows &&
    input.length().to_int64() >= required &&
    weight.length() == width &&
    output.length().to_int64() == packed &&
    (if plan.mode() is Bf16NormBf16WeightSiluGate {
      gate.length().to_int64() == packed
    } else {
      gate.is_empty()
    }) else {
    fail("invalid normalization buffers")
  }
  for row in 0..<rows {
    for vector in 0..<vectors {
      let base = (row.to_int64() * row_stride).to_int() + vector * vector_stride
      let out = (row * vectors + vector) * width
      let mut square = 0.0F
      for column in 0..<width {
        let x = round_bf16(input[base + column])
        square += x * x
      }
      let inverse = 1.0F /
        (square / Float::from_int(width) + Float::from_double(plan.epsilon())).sqrt()
      for column in 0..<width {
        let normalized = round_bf16(input[base + column]) * inverse
        let gamma = round_bf16(weight[column])
        output[out + column] = match plan.mode() {
          Weighted(DirectGamma) => round_bf16(normalized * gamma)
          Weighted(OnePlusGamma) => round_bf16(normalized * (1.0F + gamma))
          Bf16NormBf16WeightSiluGate => {
            let weighted = round_bf16(round_bf16(normalized) * gamma)
            let g = round_bf16(gate[out + column])
            round_bf16(weighted * (g / (1.0F + exp32(-g))))
          }
        }
      }
    }
  }
}
