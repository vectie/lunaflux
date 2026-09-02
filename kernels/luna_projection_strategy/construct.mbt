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
  padded_matrix_vector_tile? : ProjectionTile? = None,
  matrix_tile? : ProjectionTile? = None,
) -> ProjectionCapabilities raise ProjectionStrategyError {
  if subgroup_width <= 0 ||
    subgroup_width > 1024 ||
    (subgroup_width & (subgroup_width - 1)) != 0 {
    raise InvalidProjectionStrategy(Capability)
  }
  { subgroup_width, padded_matrix_vector_tile, matrix_tile }
}

///|
fn valid_projection_identity(value : StringView) -> Bool {
  value.length() > 0 && value.length() <= 128
}

///|
pub fn ProjectionBackendId::new(
  stable_name : String,
) -> ProjectionBackendId raise ProjectionStrategyError {
  if !valid_projection_identity(stable_name) {
    raise InvalidProjectionStrategy(Backend)
  }
  { stable_name, }
}

///|
pub fn ProjectionTargetId::new(
  stable_name : String,
) -> ProjectionTargetId raise ProjectionStrategyError {
  if !valid_projection_identity(stable_name) {
    raise InvalidProjectionStrategy(Target)
  }
  { stable_name, }
}

///|
pub fn ProjectionStrategy::new(
  stable_id~ : Int,
  phase~ : ProjectionPhase,
  strategy_class? : ProjectionStrategyClass? = None,
  kernel_class~ : ProjectionKernelClass,
  row_bucket~ : Int,
  tile~ : ProjectionTile,
) -> ProjectionStrategy raise ProjectionStrategyError {
  if stable_id <= 0 || row_bucket <= 0 {
    raise InvalidProjectionStrategy(Candidate)
  }
  if projection_row_bucket(row_bucket) != row_bucket ||
    (phase == Decode && row_bucket != 1) ||
    (phase == Prefill && row_bucket == 1) {
    raise InvalidProjectionStrategy(Bucket)
  }
  let resolved_class = match strategy_class {
    Some(value) => value
    None =>
      if phase == Decode {
        DecodeGemv
      } else if row_bucket <= 8 {
        SmallRowTiledGemm
      } else {
        LargeRowTiledGemm
      }
  }
  let valid_class = match (phase, resolved_class) {
    (Decode, DecodeGemv) => row_bucket == 1
    (Prefill, SmallRowTiledGemm) => row_bucket <= 8
    (Prefill, LargeRowTiledGemm) => row_bucket > 8
    _ => false
  }
  if !valid_class {
    raise InvalidProjectionStrategy(Candidate)
  }
  {
    stable_id,
    phase,
    strategy_class: resolved_class,
    kernel_class,
    row_bucket,
    tile,
  }
}

///|
pub fn ProjectionAutotuneKey::new(
  backend~ : ProjectionBackendId,
  target~ : ProjectionTargetId,
  numeric_type~ : ProjectionNumericType,
  shape~ : ProjectionShape,
  strategy~ : ProjectionStrategy,
) -> ProjectionAutotuneKey {
  { backend, target, numeric_type, shape, strategy }
}

///|
pub fn ProjectionAutotuneRecord::new(
  key~ : ProjectionAutotuneKey,
  median_latency_ns~ : Int64,
  sample_count~ : Int,
) -> ProjectionAutotuneRecord raise ProjectionStrategyError {
  if median_latency_ns <= 0L || sample_count < 3 {
    raise InvalidProjectionStrategy(Measurement)
  }
  { key, median_latency_ns, sample_count }
}
