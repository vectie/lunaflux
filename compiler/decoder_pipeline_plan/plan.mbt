///|
/// A nonempty contiguous interval of a complete decoder. Physical device,
/// model-family names and transport do not participate in this pure plan.
pub struct DecoderStageRange {
  priv layers : Int
  priv first : Int
  priv end : Int
}

///|
pub fn DecoderStageRange::new(
  layer_count~ : Int,
  first_layer~ : Int,
  end_layer~ : Int,
) -> Self raise @precision.PrecisionPlanError {
  guard first_layer >= 0 && end_layer > first_layer && end_layer <= layer_count else {
    raise IllegalPlacement
  }
  { layers: layer_count, first: first_layer, end: end_layer, }
}

///|
pub fn DecoderStageRange::layer_count(self : Self) -> Int {
  self.layers
}

///|
pub fn DecoderStageRange::layer_range(self : Self) -> (Int, Int) {
  (self.first, self.end)
}

///|
pub fn DecoderStageRange::text_boundary(self : Self) -> @precision.TextBoundary {
  match (self.first == 0, self.end == self.layers) {
    (true, true) => Complete
    (true, false) => Ingress
    (false, true) => Egress
    (false, false) => Interior
  }
}

///|
/// All layers occur exactly once and in order. Views cannot mutate placement.
pub struct DecoderPipelinePlan {
  priv ranges : Array[DecoderStageRange]
  priv bytes : Array[Int64]
}

///|
pub fn DecoderPipelinePlan::stages(self : Self) -> ArrayView[DecoderStageRange] {
  self.ranges[:]
}

///|
pub fn DecoderPipelinePlan::stage_bytes(self : Self) -> ArrayView[Int64] {
  self.bytes[:]
}

///|
fn add(a : Int64, b : Int64) -> Int64 raise @precision.PrecisionPlanError {
  guard a >= 0L && b >= 0L && a <= 9223372036854775807L - b else {
    raise MemoryBudgetExceeded
  }
  a + b
}

///|
/// Partition real layer footprints under each host's available ceiling. The
/// caller includes resident weights/state/workspace in layer_bytes; per-stage
/// residuals, request ports and reserve belong in stage_overhead_bytes. Global
/// ingress/egress weights belong only to first/last stages. This is a memory
/// feasibility planner, not a latency-balancing or offload performance claim.
/// Suffix dynamic programming avoids an unsafe greedy placement on unlike hosts.
pub fn DecoderPipelinePlan::partition(
  layer_bytes : ArrayView[Int64],
  device_budgets : ArrayView[Int64],
  stage_overhead_bytes~ : Int64,
  ingress_bytes~ : Int64,
  egress_bytes~ : Int64,
) -> Self raise @precision.PrecisionPlanError {
  let layers = layer_bytes.length()
  let stages = device_budgets.length()
  guard layers > 0 && stages > 0 && stages <= layers else {
    raise IllegalPlacement
  }
  let sums = [0L]
  for bytes in layer_bytes {
    sums.push(add(sums[sums.length() - 1], bytes))
  }
  let bases = device_budgets.mapi((rank, budget) => {
    guard budget > 0L else {
      raise @precision.PrecisionPlanError::MemoryBudgetExceeded
    }
    let base = add(
      stage_overhead_bytes,
      if rank == 0 {
        ingress_bytes
      } else {
        0L
      },
    )
    add(base, if rank == stages - 1 { egress_bytes } else { 0L })
  })
  let feasible = Array::makei(stages + 1, _ => {
    FixedArray::make(layers + 1, false)
  })
  feasible[stages][layers] = true
  for rank = stages - 1; rank >= 0; rank = rank - 1 {
    for first in 0..<layers {
      for end in (first + 1)..<=layers {
        if bases[rank] <= device_budgets[rank] &&
          sums[end] - sums[first] <= device_budgets[rank] - bases[rank] &&
          feasible[rank + 1][end] {
          feasible[rank][first] = true
          break
        }
      }
    }
  }
  guard feasible[0][0] else { raise MemoryBudgetExceeded }
  let ranges = []
  let bytes = []
  for rank in 0..<stages; first = 0 {
    // Deterministic largest legal prefix that still leaves a feasible suffix.
    let end = for end = layers; end > first; end = end - 1 {
      if bases[rank] <= device_budgets[rank] &&
        sums[end] - sums[first] <= device_budgets[rank] - bases[rank] &&
        feasible[rank + 1][end] {
        break end
      }
    } nobreak {
      raise IllegalPlacement
    }
    ranges.push(
      DecoderStageRange::new(
        layer_count=layers,
        first_layer=first,
        end_layer=end,
      ),
    )
    bytes.push(add(bases[rank], sums[end] - sums[first]))
    continue end
  }
  { ranges, bytes, }
}
