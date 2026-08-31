///|
const SAMPLING_PARAMETER_ROW_BYTES : Int64 = 40L

///|
const SAMPLING_RESULT_ROW_BYTES : Int64 = 8L

///|
fn checked_sampling_product(
  left : Int64,
  right : Int64,
) -> Int64 raise SamplingStrategyError {
  if left < 0L ||
    right < 0L ||
    (right != 0L && left > 9223372036854775807L / right) {
    raise InvalidSamplingStrategy(Workspace)
  }
  left * right
}

///|
/// Constructs the fixed parameter, result, and candidate-workspace geometry.
/// Candidate token IDs are I32 and weights are F64. The latter fixes the
/// accumulation precision independently of the backend's ordinary tensor dtype.
pub fn plan_device_sampling(
  maximum_rows~ : Int,
  vocabulary_size~ : Int,
  maximum_top_k~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> SamplingDevicePlan raise SamplingStrategyError {
  if maximum_rows <= 0 || vocabulary_size <= 0 {
    raise InvalidSamplingStrategy(Shape)
  }
  if maximum_top_k <= 0 || maximum_top_k > vocabulary_size {
    raise InvalidSamplingStrategy(CandidateBound)
  }
  if maximum_workspace_bytes <= 0L {
    raise InvalidSamplingStrategy(Workspace)
  }
  let candidate_count = checked_sampling_product(
    maximum_rows.to_int64(),
    maximum_top_k.to_int64(),
  )
  let token_bytes = checked_sampling_product(candidate_count, 4L)
  let weight_bytes = checked_sampling_product(candidate_count, 8L)
  if token_bytes > 9223372036854775800L {
    raise InvalidSamplingStrategy(Workspace)
  }
  let candidate_weight_offset_bytes = (token_bytes + 7L) / 8L * 8L
  if candidate_weight_offset_bytes > 9223372036854775807L - weight_bytes {
    raise InvalidSamplingStrategy(Workspace)
  }
  let workspace_bytes = candidate_weight_offset_bytes + weight_bytes
  if workspace_bytes > maximum_workspace_bytes {
    raise InvalidSamplingStrategy(Workspace)
  }
  {
    maximum_rows,
    vocabulary_size,
    maximum_top_k,
    parameter_row_bytes: SAMPLING_PARAMETER_ROW_BYTES,
    result_row_bytes: SAMPLING_RESULT_ROW_BYTES,
    candidate_token_offset_bytes: 0L,
    candidate_weight_offset_bytes,
    workspace_bytes,
  }
}

///|
/// Chooses the token-step route without allocation or backend inspection.
///
/// `mode=0` is greedy and `mode=1` is stochastic. Stochastic device execution
/// requires an explicit bounded top-k. Top-p is optional and, when present,
/// must lie in `(0, 1]`; it is applied after top-k ordering.
pub fn select_sampling_route(
  plan : SamplingDevicePlan,
  mode~ : Int,
  temperature~ : Double,
  has_top_k~ : Bool,
  top_k~ : Int,
  has_top_p~ : Bool,
  top_p~ : Double,
) -> SamplingRoute raise SamplingStrategyError {
  if mode == 0 {
    if temperature != 1.0 || has_top_k || has_top_p {
      raise InvalidSamplingStrategy(Parameters)
    }
    return DeviceGreedy
  }
  if mode != 1 ||
    temperature.is_nan() ||
    temperature.is_inf() ||
    temperature <= 0.0 ||
    (
      has_top_p &&
      (top_p.is_nan() || top_p.is_inf() || top_p <= 0.0 || top_p > 1.0)
    ) {
    raise InvalidSamplingStrategy(Parameters)
  }
  if !has_top_k {
    return HostCanonical
  }
  if top_k <= 0 {
    raise InvalidSamplingStrategy(Parameters)
  }
  if top_k > plan.maximum_top_k {
    HostCanonical
  } else {
    DeviceRestrictiveTopK
  }
}
