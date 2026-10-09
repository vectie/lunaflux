///|
fn validate_rotary_frequency(
  frequency : RotaryFrequency,
  positions : Int,
) -> Unit raise PrecisionPlanError {
  match frequency {
    Base(theta~) => {
      guard rotary_positive(theta) else { raise IllegalPlacement }
    }
    Yarn(theta~, original_positions~, factor~, beta_fast~, beta_slow~) => {
      guard theta > 1.0 &&
        rotary_positive(theta) &&
        original_positions > 0 &&
        original_positions <= positions &&
        rotary_positive(factor) &&
        rotary_positive(beta_fast) &&
        rotary_positive(beta_slow) &&
        beta_fast >= beta_slow else {
        raise IllegalPlacement
      }
    }
  }
}

///|
/// Adjacent-pair F32 rotation of a BF16 tensor's suffix. The unrotated prefix
/// is copied bitwise; inverse rotation conjugates the frequency, not the data.
pub struct RotaryTensorPrecision {
  priv heads : Int
  priv width : Int
  priv suffix : Int
  priv rows : Int
  priv positions : Int
  priv frequency : RotaryFrequency
  priv inverse : Bool
  priv bytes : Int64
}

///|
pub fn RotaryTensorPrecision::new(
  head_count~ : Int,
  head_width~ : Int,
  rotary_width~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  frequency~ : RotaryFrequency,
  inverse~ : Bool,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard head_count > 0 &&
    head_count <= 512 &&
    head_width > 0 &&
    head_width <= 2048 &&
    head_width % 2 == 0 &&
    rotary_width > 0 &&
    rotary_width <= head_width &&
    rotary_width % 2 == 0 &&
    maximum_rows > 0 &&
    maximum_rows <= 65536 &&
    maximum_positions > 0 else {
    raise IllegalPlacement
  }
  validate_rotary_frequency(frequency, maximum_positions)
  let bytes = maximum_rows.to_int64() *
    head_count.to_int64() *
    head_width.to_int64() *
    2L
  guard bytes <= maximum_workspace_bytes else { raise MemoryBudgetExceeded }
  {
    heads: head_count,
    width: head_width,
    suffix: rotary_width,
    rows: maximum_rows,
    positions: maximum_positions,
    frequency,
    inverse,
    bytes,
  }
}

///|
pub fn RotaryTensorPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int, Int) {
  (self.heads, self.width, self.suffix, self.rows, self.positions)
}

///|
pub fn RotaryTensorPrecision::frequency(self : Self) -> RotaryFrequency {
  self.frequency
}

///|
pub fn RotaryTensorPrecision::inverse(self : Self) -> Bool {
  self.inverse
}

///|
pub fn RotaryTensorPrecision::workspace_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn RotaryTensorPrecision::effect(_self : Self) -> NumericEffect {
  StepWrite
}
