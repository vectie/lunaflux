///|
pub fn ProjectionShape::new(
  family : ProjectionSemanticFamily,
  input_width~ : Int,
  output_width~ : Int,
  intermediate_width? : Int = 0,
) -> ProjectionShape raise ProjectionStrategyError {
  if input_width <= 0 || output_width <= 0 || intermediate_width < 0 {
    raise InvalidProjectionStrategy(Shape)
  }
  if family == GatedMlp && intermediate_width == 0 {
    raise InvalidProjectionStrategy(Shape)
  }
  { family, input_width, output_width, intermediate_width }
}

///|
pub fn ProjectionTile::new(
  rows~ : Int,
  columns~ : Int,
  reduction~ : Int,
) -> ProjectionTile raise ProjectionStrategyError {
  if rows <= 0 || columns <= 0 || reduction <= 0 {
    raise InvalidProjectionStrategy(Shape)
  }
  { rows, columns, reduction }
}

///|
pub fn ProjectionCapabilities::new(
  subgroup_width~ : Int,
  matrix_tile? : ProjectionTile? = None,
) -> ProjectionCapabilities raise ProjectionStrategyError {
  if subgroup_width <= 0 ||
    subgroup_width > 1024 ||
    (subgroup_width & (subgroup_width - 1)) != 0 {
    raise InvalidProjectionStrategy(Capability)
  }
  { subgroup_width, matrix_tile }
}

///|
pub fn ProjectionStrategy::new(
  stable_id~ : Int,
  phase~ : ProjectionPhase,
  kernel_class~ : ProjectionKernelClass,
  row_bucket~ : Int,
  tile~ : ProjectionTile,
) -> ProjectionStrategy raise ProjectionStrategyError {
  if stable_id <= 0 || row_bucket <= 0 {
    raise InvalidProjectionStrategy(Candidate)
  }
  if phase == Decode && row_bucket != 1 {
    raise InvalidProjectionStrategy(Bucket)
  }
  { stable_id, phase, kernel_class, row_bucket, tile }
}

///|
pub fn ProjectionAutotuneRecord::new(
  device_class~ : String,
  numeric_format~ : String,
  shape~ : ProjectionShape,
  strategy~ : ProjectionStrategy,
  median_latency_ns~ : Int64,
  sample_count~ : Int,
) -> ProjectionAutotuneRecord raise ProjectionStrategyError {
  if device_class.length() == 0 ||
    numeric_format.length() == 0 ||
    median_latency_ns <= 0L ||
    sample_count < 3 {
    raise InvalidProjectionStrategy(Measurement)
  }
  {
    device_class,
    numeric_format,
    shape,
    strategy,
    median_latency_ns,
    sample_count,
  }
}
