///|
/// BF16 shared sigmoid/product before adding the FP32 routed buffer.
pub fn shared_expert_into(
  plan : @precision.SharedExpertPrecision,
  routed : ArrayView[Float],
  shared : ArrayView[Float],
  gate_logits : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (width, maximum_rows) = plan.geometry()
  guard rows > 0 &&
    rows <= maximum_rows &&
    routed.length().to_int64() == rows.to_int64() * width.to_int64() &&
    shared.length() == routed.length() &&
    output.length() == routed.length() &&
    gate_logits.length() == rows else {
    fail("invalid shared expert buffers")
  }
  for row in 0..<rows {
    let gate = sigmoid_bf16(gate_logits[row])
    for column in 0..<width {
      let cell = row * width + column
      let weighted = round_bf16(round_bf16(shared[cell]) * gate)
      output[cell] = round_bf16(routed[cell] + weighted)
    }
  }
}
