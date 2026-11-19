///|
pub(all) enum DeltaDecayLayout {
  PerKeyComponent
  PerValueHead
} derive(Debug, Eq)

///|
pub extend DeltaDecayLayout with @debug.Debug::{to_repr}

///|
pub extend DeltaDecayLayout with Eq::{equal, not_equal}

///|
/// Ordered normalized delta attention. Sequence ordinals address a frame;
/// persistent cache slots address requests and are independent of batch order.
/// Inputs/output are BF16; normalization, decay and persistent state are F32.
pub struct RecurrentDeltaPrecision {
  priv heads : Int32
  priv query_key_heads : Int32
  priv decay_layout : DeltaDecayLayout
  priv query_key_row_stride : Int64
  priv value_row_stride : Int64
  priv keys : Int32
  priv values : Int32
  priv rows : Int32
  priv sequences : Int32
  priv slots : Int32
  priv epsilon : Double
  priv decay_lower : Double
  priv state_bytes : Int64
  priv evaluation : RecurrentEvaluationLaw
}

///|
pub fn RecurrentDeltaPrecision::new(
  head_count~ : Int32,
  key_dimension~ : Int32,
  value_dimension~ : Int32,
  maximum_rows~ : Int32,
  maximum_sequences~ : Int32,
  cache_slots~ : Int32,
  normalization_epsilon~ : Double,
  maximum_state_bytes~ : Int64,
  log_decay_lower_bound~ : Double = -1.0 / 0.0,
  query_key_heads~ : Int32 = head_count,
  decay_layout~ : DeltaDecayLayout = PerKeyComponent,
  query_key_row_stride? : Int64,
  value_row_stride? : Int64,
  evaluation~ : RecurrentEvaluationLaw = OrderedTokenFloat32,
) -> Self raise PrecisionPlanError {
  guard head_count > 0 &&
    query_key_heads > 0 &&
    head_count % query_key_heads == 0 &&
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
    log_decay_lower_bound < 0.0 else {
    raise MissingAccuracyContract
  }
  let qk_width = expert_checked_mul(
    query_key_heads.to_int64(),
    key_dimension.to_int64(),
  )
  let value_width = expert_checked_mul(
    head_count.to_int64(),
    value_dimension.to_int64(),
  )
  let qk_stride = query_key_row_stride.unwrap_or(qk_width)
  let value_stride = value_row_stride.unwrap_or(value_width)
  guard qk_stride >= qk_width && value_stride >= value_width else {
    raise IllegalPlacement
  }
  // The triangular lowering currently owns only the original equal-head,
  // per-key layout. Ordinary token recurrence supports both layouts.
  guard evaluation is OrderedTokenFloat32 ||
    (
      query_key_heads == head_count &&
      decay_layout is PerKeyComponent &&
      qk_stride == qk_width &&
      value_stride == value_width
    ) else {
    raise NoImplementation
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
        expert_checked_mul(maximum_rows.to_int64(), query_key_heads.to_int64()),
        key_dimension.to_int64(),
      ),
      4L,
    ),
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(maximum_rows.to_int64(), head_count.to_int64()),
      if decay_layout is PerKeyComponent {
        key_dimension.to_int64() * 4L
      } else {
        4L
      },
    ),
  )
  ignore(
    expert_checked_mul(
      maximum_rows.to_int64(),
      expert_checked_mul(qk_stride, 2L),
    ),
  )
  ignore(
    expert_checked_mul(
      maximum_rows.to_int64(),
      expert_checked_mul(value_stride, 2L),
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
    query_key_heads,
    decay_layout,
    query_key_row_stride: qk_stride,
    value_row_stride: value_stride,
    keys: key_dimension,
    values: value_dimension,
    rows: maximum_rows,
    sequences: maximum_sequences,
    slots: cache_slots,
    epsilon: normalization_epsilon,
    decay_lower: log_decay_lower_bound,
    state_bytes,
    evaluation,
  }
}

///|
pub fn RecurrentDeltaPrecision::geometry(self : Self) -> (Int32, Int32, Int32) {
  (self.heads, self.keys, self.values)
}

///|
pub fn RecurrentDeltaPrecision::query_key_head_count(self : Self) -> Int32 {
  self.query_key_heads
}

///|
pub fn RecurrentDeltaPrecision::decay_layout(self : Self) -> DeltaDecayLayout {
  self.decay_layout
}

///|
/// Strides are BF16 element counts. Q and K use separate pointers and the same
/// row stride; V has its own stride. Output rows are always packed.
pub fn RecurrentDeltaPrecision::query_key_row_stride(self : Self) -> Int64 {
  self.query_key_row_stride
}

///|
pub fn RecurrentDeltaPrecision::value_row_stride(self : Self) -> Int64 {
  self.value_row_stride
}

///|
pub fn RecurrentDeltaPrecision::capacity(self : Self) -> (Int32, Int32, Int32) {
  (self.rows, self.sequences, self.slots)
}

///|
pub fn RecurrentDeltaPrecision::normalization_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
/// Negative infinity denotes an unbounded non-positive input domain. This is
/// a numerical input contract, not an instruction to clamp model decay.
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

///|
pub fn RecurrentDeltaPrecision::evaluation(
  self : Self,
) -> RecurrentEvaluationLaw {
  self.evaluation
}
