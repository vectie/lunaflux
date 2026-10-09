///|
/// Frequency policy is semantic; neither branch describes a device schedule.
pub(all) enum RotaryFrequency {
  Base(theta~ : Double)
  Yarn(
    theta~ : Double,
    original_positions~ : Int,
    factor~ : Double,
    beta_fast~ : Double,
    beta_slow~ : Double
  )
} derive(Debug, Eq)

///|
pub extend RotaryFrequency with Eq::{equal, not_equal}

///|
pub extend RotaryFrequency with @debug.Debug::{to_repr}

///|
/// Quantize/dequantize into BF16, not a change to persistent cache storage.
pub(all) enum SimulatedCacheScale {
  LinearF32
  PowerOfTwo
} derive(Debug, Eq)

///|
pub extend SimulatedCacheScale with Eq::{equal, not_equal}

///|
pub extend SimulatedCacheScale with @debug.Debug::{to_repr}

///|
/// Full-head adjacent-pair F32 rotary, BF16 publication, then blockwise
/// E4M3 simulation of KV's non-rotary prefix. The rotary suffix is excluded
/// from quantization. The cache owner commits these outputs separately.
pub struct RotaryKvPrecision {
  priv query_heads : Int
  priv kv_heads : Int
  priv width : Int
  priv rotary_width : Int
  priv rows : Int
  priv positions : Int
  priv frequency : RotaryFrequency
  priv block_width : Int
  priv amax_floor : Double
  priv scale : SimulatedCacheScale
  priv frames : ReadOnlyArray[Int64]
}

///|
fn rotary_positive(value : Double) -> Bool {
  value > 0.0 && !value.is_nan() && !value.is_inf()
}

///|
fn validate_cache_simulation(
  width : Int,
  rotary : Int,
  block : Int,
  floor : Double,
) -> Unit raise PrecisionPlanError {
  guard block > 0 &&
    block <= 1024 &&
    (block & (block - 1)) == 0 &&
    (width - rotary) % block == 0 &&
    rotary_positive(floor) &&
    floor >= 1.0e-30 &&
    floor <= 1.0 else {
    raise IllegalPlacement
  }
}

///|
pub fn RotaryKvPrecision::new(
  query_heads~ : Int,
  key_value_heads~ : Int,
  head_width~ : Int,
  rotary_width~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  frequency~ : RotaryFrequency,
  simulation_block_width~ : Int,
  simulation_amax_floor~ : Double,
  simulation_scale~ : SimulatedCacheScale,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard query_heads > 0 &&
    query_heads <= 512 &&
    key_value_heads > 0 &&
    key_value_heads <= 512 &&
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
  validate_cache_simulation(
    head_width, rotary_width, simulation_block_width, simulation_amax_floor,
  )
  validate_rotary_frequency(frequency, maximum_positions)
  let bytes = (heads : Int) => {
    maximum_rows.to_int64() * heads.to_int64() * head_width.to_int64() * 2L
  }
  let frames = [bytes(query_heads), bytes(key_value_heads)]
  guard frames[0] + frames[1] <= maximum_workspace_bytes else {
    raise MemoryBudgetExceeded
  }
  {
    query_heads,
    kv_heads: key_value_heads,
    width: head_width,
    rotary_width,
    rows: maximum_rows,
    positions: maximum_positions,
    frequency,
    block_width: simulation_block_width,
    amax_floor: simulation_amax_floor,
    scale: simulation_scale,
    frames: ReadOnlyArray::from_array(frames),
  }
}

///|
pub fn RotaryKvPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int, Int, Int) {
  (
    self.query_heads,
    self.kv_heads,
    self.width,
    self.rotary_width,
    self.rows,
    self.positions,
  )
}

///|
pub fn RotaryKvPrecision::frequency(self : Self) -> RotaryFrequency {
  self.frequency
}

///|
pub fn RotaryKvPrecision::simulation(
  self : Self,
) -> (Int, Double, SimulatedCacheScale) {
  (self.block_width, self.amax_floor, self.scale)
}

///|
pub fn RotaryKvPrecision::frame_bytes(self : Self) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn RotaryKvPrecision::workspace_bytes(self : Self) -> Int64 {
  self.frames[0] + self.frames[1]
}

///|
pub fn RotaryKvPrecision::effect(_self : Self) -> NumericEffect {
  StepWrite
}
