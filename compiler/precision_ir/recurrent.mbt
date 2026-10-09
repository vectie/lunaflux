///|
/// Ordered normalized delta attention. Sequence ordinals address a frame;
/// persistent cache slots address requests and are independent of batch order.
/// Inputs/output are BF16; normalization, decay and persistent state are F32.
pub struct RecurrentDeltaPrecision {
  priv heads : Int
  priv keys : Int
  priv values : Int
  priv rows : Int
  priv sequences : Int
  priv slots : Int
  priv epsilon : Double
  priv decay_lower : Double
  priv state_bytes : Int64
}

///|
pub fn RecurrentDeltaPrecision::new(
  head_count~ : Int,
  key_dimension~ : Int,
  value_dimension~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  normalization_epsilon~ : Double,
  log_decay_lower_bound~ : Double,
  maximum_state_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard head_count > 0 &&
    key_dimension > 0 &&
    value_dimension > 0 &&
    maximum_rows > 0 &&
    maximum_sequences > 0 &&
    maximum_sequences <= maximum_rows &&
    cache_slots >= maximum_sequences else {
    raise IllegalPlacement
  }
  guard normalization_epsilon > 0.0 &&
    !normalization_epsilon.is_inf() &&
    log_decay_lower_bound < 0.0 &&
    !log_decay_lower_bound.is_inf() else {
    raise MissingAccuracyContract
  }
  let state_bytes = expert_checked_mul(
    expert_checked_mul(
      expert_checked_mul(
        expert_checked_mul(cache_slots.to_int64(), head_count.to_int64()),
        key_dimension.to_int64(),
      ),
      value_dimension.to_int64(),
    ),
    4L,
  )
  guard state_bytes <= maximum_state_bytes else { raise MemoryBudgetExceeded }
  ignore(
    expert_checked_mul(
      expert_checked_mul(
        expert_checked_mul(maximum_rows.to_int64(), head_count.to_int64()),
        key_dimension.to_int64(),
      ),
      4L,
    ),
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(
        expert_checked_mul(maximum_rows.to_int64(), head_count.to_int64()),
        value_dimension.to_int64(),
      ),
      2L,
    ),
  )
  {
    heads: head_count,
    keys: key_dimension,
    values: value_dimension,
    rows: maximum_rows,
    sequences: maximum_sequences,
    slots: cache_slots,
    epsilon: normalization_epsilon,
    decay_lower: log_decay_lower_bound,
    state_bytes,
  }
}

///|
pub fn RecurrentDeltaPrecision::geometry(self : Self) -> (Int, Int, Int) {
  (self.heads, self.keys, self.values)
}

///|
pub fn RecurrentDeltaPrecision::capacity(self : Self) -> (Int, Int, Int) {
  (self.rows, self.sequences, self.slots)
}

///|
pub fn RecurrentDeltaPrecision::normalization_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn RecurrentDeltaPrecision::log_decay_lower_bound(self : Self) -> Double {
  self.decay_lower
}

///|
pub fn RecurrentDeltaPrecision::state_bytes(self : Self) -> Int64 {
  self.state_bytes
}

///|
pub fn RecurrentDeltaPrecision::effect(self : Self) -> NumericEffect {
  ignore(self)
  CacheCommit
}
