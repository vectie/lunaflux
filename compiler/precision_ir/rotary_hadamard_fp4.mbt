///|
/// Suffix rotary -> normalized full-vector Sylvester Hadamard -> block E2M1
/// quantize/dequantize. Both intermediate publications round to BF16; the
/// final storage remains BF16, not packed FP4. No device geometry is semantic.
pub struct RotaryHadamardFp4Precision {
  priv rotary : RotaryTensorPrecision
  priv block : Int
  priv frames : ReadOnlyArray[Int64]
}

///|
pub fn RotaryHadamardFp4Precision::new(
  head_count~ : Int,
  head_width~ : Int,
  rotary_width~ : Int,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  frequency~ : RotaryFrequency,
  simulation_block_width~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard head_width > 0 &&
    (head_width & (head_width - 1)) == 0 &&
    simulation_block_width > 0 &&
    simulation_block_width <= head_width &&
    (simulation_block_width & (simulation_block_width - 1)) == 0 &&
    head_width % simulation_block_width == 0 else {
    raise IllegalPlacement
  }
  let rotary = RotaryTensorPrecision::new(
    head_count~,
    head_width~,
    rotary_width~,
    maximum_rows~,
    maximum_positions~,
    frequency~,
    inverse=false,
    maximum_workspace_bytes~,
  )
  let frames = [rotary.workspace_bytes(), rotary.workspace_bytes()]
  guard frames[0] + frames[1] <= maximum_workspace_bytes else {
    raise MemoryBudgetExceeded
  }
  {
    rotary,
    block: simulation_block_width,
    frames: ReadOnlyArray::from_array(frames),
  }
}

///|
pub fn RotaryHadamardFp4Precision::rotary(self : Self) -> RotaryTensorPrecision {
  self.rotary
}

///|
pub fn RotaryHadamardFp4Precision::block_width(self : Self) -> Int {
  self.block
}

///|
/// The reference passes width^-0.5 as one scalar; device approximate reciprocal
/// square root must not change that constant at a BF16 midpoint.
pub fn RotaryHadamardFp4Precision::normalization(self : Self) -> Double {
  let (_, width, _, _, _) = self.rotary.geometry()
  1.0 / width.to_double().sqrt()
}

///|
pub fn RotaryHadamardFp4Precision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn RotaryHadamardFp4Precision::workspace_bytes(self : Self) -> Int64 {
  self.frames[0] + self.frames[1]
}

///|
pub fn RotaryHadamardFp4Precision::effect(_self : Self) -> NumericEffect {
  StepWrite
}
