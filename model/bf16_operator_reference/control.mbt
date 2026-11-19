///|
/// Offline control reference. Inputs round to BF16; logarithmic decay retains
/// FP32 and may be below -5. Exp/log1p are host primitives, not GPU bit oracles.
pub fn gated_delta_control_into(
  plan : @precision.GatedDeltaControlPrecision,
  a : ArrayView[Float],
  b : ArrayView[Float],
  a_log : ArrayView[Float],
  dt_bias : ArrayView[Float],
  log_decay : FixedArray[Float],
  beta : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (heads, maximum_rows) = plan.geometry()
  let elements = heads.to_int64() * rows.to_int64()
  guard rows > 0 &&
    rows <= maximum_rows &&
    a.length().to_int64() == elements &&
    b.length() == a.length() &&
    log_decay.length() == a.length() &&
    beta.length() == a.length() &&
    a_log.length() == heads &&
    dt_bias.length() == heads else {
    fail("invalid control buffers")
  }
  for row in 0..<rows {
    for head in 0..<heads {
      let index = row * heads + head
      let biased = round_bf16(a[index]) + round_bf16(dt_bias[head])
      let softplus = if biased > 20.0F {
        biased
      } else {
        @math.ln_1pf(exp32(biased))
      }
      log_decay[index] = -exp32(round_bf16(a_log[head])) * softplus
      beta[index] = round_bf16(1.0F / (1.0F + exp32(-round_bf16(b[index]))))
    }
  }
}
