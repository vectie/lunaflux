///|
/// Immutable selected-expert compute plan. All weight operands are decoded at
/// consumption; BF16 activation/product rounding is explicit. Rank-local
/// contributions are combined in F32 so the same plan can feed an all-reduce.
pub struct ExpertMlpPrecision {
  priv weights : ReadOnlyArray[OperandPlan]
  priv local_experts : ReadOnlyArray[Int]
  priv expert_map : ReadOnlyArray[Int]
  priv expert_count : Int
  priv selected_experts : Int
  priv maximum_rows : Int
  priv hidden : Int
  priv intermediate : Int
  priv clamp : Double?
  priv offsets : ReadOnlyArray[Int64]
  priv stride : Int64
  priv bank_bytes : Int64
  priv intermediate_bytes : Int64
  priv weighted_bytes : Int64
  priv total_bytes : Int64
}

///|
fn expert_checked_mul(a : Int64, b : Int64) -> Int64 raise PrecisionPlanError {
  if a < 0L || b < 0L || (b > 0L && a > 9223372036854775807L / b) {
    raise ArithmeticOverflow
  }
  a * b
}

///|
pub fn ExpertMlpPrecision::new(
  gate~ : OperandPlan,
  up~ : OperandPlan,
  down~ : OperandPlan,
  expert_count~ : Int,
  local_experts~ : ArrayView[Int],
  selected_experts~ : Int,
  maximum_rows~ : Int,
  clamp? : Double,
  memory_ceiling~ : Int64,
) -> Self raise PrecisionPlanError {
  if memory_ceiling < 0L ||
    expert_count <= 0 ||
    local_experts.is_empty() ||
    local_experts.length() > expert_count ||
    selected_experts <= 0 ||
    selected_experts > expert_count ||
    maximum_rows <= 0 {
    raise IllegalPlacement
  }
  if clamp is Some(limit) && (limit.is_nan() || limit.is_inf() || limit <= 0.0) {
    raise MissingAccuracyContract
  }
  let weights = [gate, up, down]
  for weight in weights {
    if weight.role() != Weight ||
      weight.placement() != DecodeInOperandTile ||
      weight.effect() != Pure {
      raise IllegalPlacement
    }
  }
  let (intermediate, hidden) = gate.representation().shape()
  if up.representation().shape() != (intermediate, hidden) ||
    down.representation().shape() != (hidden, intermediate) {
    raise IllegalPlacement
  }
  let expert_map = Array::make(expert_count, -1)
  for ordinal, expert in local_experts {
    if expert < 0 || expert >= expert_count || expert_map[expert] != -1 {
      raise IllegalPlacement
    }
    expert_map[expert] = ordinal
  }
  let offsets = []
  let mut stride = 0L
  for weight in weights {
    if stride > 9223372036854775792L {
      raise ArithmeticOverflow
    }
    stride = (stride + 15L) / 16L * 16L
    offsets.push(stride)
    let bytes = packed_buffer_layout(weight.representation()).buffer_bytes()
    if bytes > 9223372036854775807L - stride {
      raise ArithmeticOverflow
    }
    stride += bytes
  }
  if stride > 9223372036854775792L {
    raise ArithmeticOverflow
  }
  stride = (stride + 15L) / 16L * 16L
  let bank_bytes = expert_checked_mul(stride, local_experts.length().to_int64())
  let jobs = expert_checked_mul(
    maximum_rows.to_int64(),
    selected_experts.to_int64(),
  )
  let intermediate_bytes = expert_checked_mul(
    expert_checked_mul(jobs, intermediate.to_int64()),
    2L,
  )
  let weighted_bytes = expert_checked_mul(
    expert_checked_mul(jobs, hidden.to_int64()),
    2L,
  )
  let row_cells = expert_checked_mul(maximum_rows.to_int64(), hidden.to_int64())
  let sizes = [
    bank_bytes,
    intermediate_bytes,
    weighted_bytes,
    expert_checked_mul(row_cells, 2L),
    expert_checked_mul(row_cells, 4L),
    expert_checked_mul(jobs, 8L),
    expert_checked_mul(expert_count.to_int64(), 4L),
    20L,
  ]
  let mut total_bytes = 0L
  for size in sizes {
    if size > memory_ceiling - total_bytes {
      raise MemoryBudgetExceeded
    }
    total_bytes += size
  }
  {
    weights: ReadOnlyArray::from_array(weights),
    local_experts: ReadOnlyArray::from_array(local_experts),
    expert_map: ReadOnlyArray::from_array(expert_map),
    expert_count,
    selected_experts,
    maximum_rows,
    hidden,
    intermediate,
    clamp,
    offsets: ReadOnlyArray::from_array(offsets),
    stride,
    bank_bytes,
    intermediate_bytes,
    weighted_bytes,
    total_bytes,
  }
}

///|
pub fn ExpertMlpPrecision::weights(self : Self) -> ReadOnlyArray[OperandPlan] {
  self.weights
}

///|
pub fn ExpertMlpPrecision::local_experts(self : Self) -> ReadOnlyArray[Int] {
  self.local_experts
}

///|
pub fn ExpertMlpPrecision::expert_map(self : Self) -> ReadOnlyArray[Int] {
  self.expert_map
}

///|
pub fn ExpertMlpPrecision::geometry(self : Self) -> (Int, Int, Int, Int, Int) {
  (
    self.maximum_rows,
    self.hidden,
    self.intermediate,
    self.expert_count,
    self.selected_experts,
  )
}

///|
pub fn ExpertMlpPrecision::clamp(self : Self) -> Double? {
  self.clamp
}

///|
pub fn ExpertMlpPrecision::weight_offsets(self : Self) -> ReadOnlyArray[Int64] {
  self.offsets
}

///|
pub fn ExpertMlpPrecision::expert_stride(self : Self) -> Int64 {
  self.stride
}

///|
pub fn ExpertMlpPrecision::bank_bytes(self : Self) -> Int64 {
  self.bank_bytes
}

///|
pub fn ExpertMlpPrecision::workspace_bytes(self : Self) -> (Int64, Int64) {
  (self.intermediate_bytes, self.weighted_bytes)
}

///|
pub fn ExpertMlpPrecision::total_bytes(self : Self) -> Int64 {
  self.total_bytes
}
