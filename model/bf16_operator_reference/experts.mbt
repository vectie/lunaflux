///|
/// Prepared BF16 expert reference. The input checkpoint layout is packed
/// [expert,gate-then-up,intermediate,hidden], down [expert,hidden,intermediate].
/// This offline ordered-dot reference does not emulate CUDA reduction trees.
pub struct Experts {
  priv plan : @precision.ExpertMlpPrecision
  priv product : FixedArray[Float]
}

///|
pub fn Experts::new(plan : @precision.ExpertMlpPrecision) -> Self raise {
  let (_, hidden, intermediate, experts, _) = plan.geometry()
  guard plan.weights().all(w => w.representation().format() is Bf16) &&
    plan.local_experts().length() == experts &&
    plan.clamp() is None &&
    plan.activation_rounding() is Bf16Stages &&
    plan.routing_placement() is AfterDownOutputRound &&
    plan.down_publication() is LocalBFloat16 &&
    plan.activation_quantization() is None &&
    hidden.to_int64() * intermediate.to_int64() * experts.to_int64() <=
    1073741823L else {
    fail("BF16 reference requires complete local staged BF16 experts")
  }
  { plan, product: FixedArray::make(intermediate, 0.0F), }
}

///|
/// Route weights use FP32 storage. The caller supplies BF16-rounded routing
/// values for MiniCPM. Scratch and output are caller-owned; no run allocations.
pub fn Experts::run_into(
  self : Self,
  input : ArrayView[Float],
  indices : ArrayView[Int32],
  routes : ArrayView[Float],
  gate_up : ArrayView[Float],
  down : ArrayView[Float],
  output : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (maximum_rows, hidden, intermediate, experts, selected) = self.plan.geometry()
  let matrix = hidden * intermediate
  guard rows > 0 &&
    rows <= maximum_rows &&
    input.length().to_int64() == rows.to_int64() * hidden.to_int64() &&
    output.length() == input.length() &&
    indices.length().to_int64() == rows.to_int64() * selected.to_int64() &&
    routes.length() == indices.length() &&
    gate_up.length().to_int64() == matrix.to_int64() * experts.to_int64() * 2L &&
    down.length().to_int64() == matrix.to_int64() * experts.to_int64() else {
    fail("invalid BF16 expert buffers")
  }
  for row in 0..<rows {
    let selection = row * selected
    let base = row * hidden
    for slot in 0..<selected {
      let expert = indices[selection + slot]
      guard expert >= 0 && expert < experts else {
        fail("invalid selected expert")
      }
      for previous in 0..<slot {
        guard indices[selection + previous] != expert else {
          fail("duplicate selected expert")
        }
      }
    }
    for column in 0..<hidden {
      output[base + column] = 0.0F
    }
    for expert in 0..<experts {
      let mut slot = -1
      for index in 0..<selected {
        if indices[selection + index] == expert {
          slot = index
        }
      }
      if slot < 0 {
        continue
      }
      let expert_base = expert * matrix * 2
      swiglu_product_into(
        input[base:base + hidden],
        gate_up[expert_base:expert_base + matrix],
        gate_up[expert_base + matrix:expert_base + matrix * 2],
        self.product,
      )
      let down_base = expert * matrix
      for column in 0..<hidden {
        let weight_base = down_base + column * intermediate
        let sum = down_dot(
          self.product[:],
          down[weight_base:weight_base + intermediate],
        )
        let weighted = round_bf16(sum * routes[selection + slot])
        let added = output[base + column] + weighted
        output[base + column] = match self.plan.combine_rounding() {
          Float32Combine => added
          Bf16EachAdd => round_bf16(added)
        }
      }
    }
  }
}
