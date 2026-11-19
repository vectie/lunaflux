///|
fn swiglu_product_into(
  input : ArrayView[Float],
  gate : ArrayView[Float],
  up : ArrayView[Float],
  product : FixedArray[Float],
) -> Unit {
  let hidden = input.length()
  for inner in 0..<product.length() {
    let base = inner * hidden
    let mut gate_sum = 0.0F
    let mut up_sum = 0.0F
    for column in 0..<hidden {
      let value = round_bf16(input[column])
      gate_sum += value * round_bf16(gate[base + column])
      up_sum += value * round_bf16(up[base + column])
    }
    let g = round_bf16(gate_sum)
    let activated = round_bf16(g / (1.0F + exp32(-g)))
    product[inner] = round_bf16(activated * round_bf16(up_sum))
  }
}

///|
fn down_dot(product : ArrayView[Float], weight : ArrayView[Float]) -> Float {
  let mut sum = 0.0F
  for inner in 0..<product.length() {
    sum += product[inner] * round_bf16(weight[inner])
  }
  round_bf16(sum)
}

///|
/// Offline staged BF16 dense/shared MLP; scratch is prepared once. Ordered
/// scalar dots establish stage boundaries, not GPU bitwise equivalence.
pub struct DenseMlp {
  priv plan : @precision.DenseSwiGluPrecision
  priv product : FixedArray[Float]
}

///|
pub fn DenseMlp::new(plan : @precision.DenseSwiGluPrecision) -> Self raise {
  let (hidden, intermediate, _) = plan.geometry()
  guard plan.publication() is LocalBFloat16 &&
    hidden.to_int64() * intermediate.to_int64() <= 2147483647L else {
    fail("dense reference requires local BF16 output")
  }
  { plan, product: FixedArray::make(intermediate, 0.0F), }
}

///|
pub fn DenseMlp::run_into(
  self : Self,
  input : ArrayView[Float],
  gate : ArrayView[Float],
  up : ArrayView[Float],
  down : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (hidden, intermediate, maximum_rows) = self.plan.geometry()
  let matrix = hidden * intermediate
  guard rows > 0 &&
    rows <= maximum_rows &&
    input.length().to_int64() == rows.to_int64() * hidden.to_int64() &&
    output.length() == input.length() &&
    gate.length() == matrix &&
    up.length() == matrix &&
    down.length() == matrix else {
    fail("invalid dense reference buffers")
  }
  for row in 0..<rows {
    let base = row * hidden
    swiglu_product_into(input[base:base + hidden], gate, up, self.product)
    for column in 0..<hidden {
      output[base + column] = down_dot(
        self.product[:],
        down[column * intermediate:(column + 1) * intermediate],
      )
    }
  }
}
