///|
/// A standalone BF16 tensor rotation followed by blockwise E4M3 simulation
/// of its non-rotary prefix. This does not commit a persistent cache or encode
/// FP8 storage. The semantic plan can consume compact pooled rows directly.
pub struct RotaryCachePrecision {
  priv rotary : RotaryTensorPrecision
  priv block : Int32
  priv floor : Double
  priv scale : SimulatedCacheScale
}

///|
pub fn RotaryCachePrecision::new(
  head_count~ : Int32,
  head_width~ : Int32,
  rotary_width~ : Int32,
  maximum_rows~ : Int32,
  maximum_positions~ : Int32,
  frequency~ : RotaryFrequency,
  simulation_block_width~ : Int32,
  simulation_amax_floor~ : Double,
  simulation_scale~ : SimulatedCacheScale,
  maximum_workspace_bytes~ : Int64,
  product~ : RotaryProductLaw = SeparateFloat32Products,
) -> Self raise PrecisionPlanError {
  let rotary = RotaryTensorPrecision::new(
    head_count~,
    head_width~,
    rotary_width~,
    maximum_rows~,
    maximum_positions~,
    frequency~,
    inverse=false,
    maximum_workspace_bytes~,
    product~,
  )
  validate_cache_simulation(
    head_width, rotary_width, simulation_block_width, simulation_amax_floor,
  )
  {
    rotary,
    block: simulation_block_width,
    floor: simulation_amax_floor,
    scale: simulation_scale,
  }
}

///|
pub fn RotaryCachePrecision::geometry(
  self : Self,
) -> (Int32, Int32, Int32, Int32, Int32) {
  self.rotary.geometry()
}

///|
pub fn RotaryCachePrecision::rotary(self : Self) -> RotaryTensorPrecision {
  self.rotary
}

///|
pub fn RotaryCachePrecision::frequency(self : Self) -> RotaryFrequency {
  self.rotary.frequency()
}

///|
pub fn RotaryCachePrecision::simulation(
  self : Self,
) -> (Int32, Double, SimulatedCacheScale) {
  (self.block, self.floor, self.scale)
}

///|
pub fn RotaryCachePrecision::workspace_bytes(self : Self) -> Int64 {
  self.rotary.workspace_bytes()
}

///|
pub fn RotaryCachePrecision::effect(_self : Self) -> NumericEffect {
  StepWrite
}
