///|
/// Terminal backend launch geometry, not an algorithm-level CUDA dependency.
pub struct ConversionLaunch {
  priv grid : Int
  priv block : Int
} derive(Debug, Eq)

///|
pub extend ConversionLaunch with Eq::{equal, not_equal}

///|
pub extend ConversionLaunch with @debug.Debug::{to_repr}

///|
fn launch(grid : Int64, block : Int) -> ConversionLaunch raise LoweringError {
  if grid <= 0L || grid > 2147483647L || block <= 0 || block > 1024 {
    raise UnsupportedGeometry
  }
  { grid: grid.to_int(), block, }
}

///|
pub fn ConversionLaunch::dimensions(self : ConversionLaunch) -> (Int, Int) {
  (self.grid, self.block)
}

///|
pub fn ConversionSource::launch(self : ConversionSource) -> ConversionLaunch {
  self.launch
}

///|
pub fn QuantizationSource::launches(
  self : QuantizationSource,
) -> ReadOnlyArray[ConversionLaunch] {
  self.launches
}
