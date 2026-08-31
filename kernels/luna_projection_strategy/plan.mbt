///|
pub fn projection_row_bucket(
  maximum_rows : Int,
) -> Int raise ProjectionStrategyError {
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidProjectionStrategy(Bucket)
  }
  let mut bucket = 1
  while bucket < maximum_rows {
    bucket *= 2
  }
  bucket
}

///|
fn projection_matrix_eligible(
  shape : ProjectionShape,
  tile : ProjectionTile,
) -> Bool {
  let (_, columns, reduction) = tile.dimensions()
  let (input_width, output_width, intermediate_width) = shape.dimensions()
  input_width % reduction == 0 &&
  output_width % columns == 0 &&
  (shape.family() != GatedMlp || intermediate_width % columns == 0)
}

///|
pub fn plan_projection_dispatch(
  shape : ProjectionShape,
  maximum_prefill_rows : Int,
  capabilities : ProjectionCapabilities,
) -> ProjectionDispatchPlan raise ProjectionStrategyError {
  let row_bucket = projection_row_bucket(maximum_prefill_rows)
  let subgroup = capabilities.subgroup_width()
  let decode_class = if shape.input_width % subgroup == 0 {
    SubgroupGemv
  } else {
    ScalarTile
  }
  let decode = ProjectionStrategy::new(
    stable_id=if decode_class == SubgroupGemv { 101 } else { 100 },
    phase=Decode,
    kernel_class=decode_class,
    row_bucket=1,
    tile=ProjectionTile::new(rows=1, columns=subgroup, reduction=subgroup),
  )
  let (prefill_class, prefill_tile, stable_id) = match
    capabilities.matrix_tile() {
    Some(tile) if maximum_prefill_rows > 1 &&
      projection_matrix_eligible(shape, tile) => (MatrixTile, tile, 201)
    _ =>
      (
        ScalarTile,
        ProjectionTile::new(rows=1, columns=subgroup, reduction=1),
        200,
      )
  }
  let prefill = ProjectionStrategy::new(
    stable_id~,
    phase=Prefill,
    kernel_class=prefill_class,
    row_bucket~,
    tile=prefill_tile,
  )
  { shape, decode, prefill }
}

///|
/// Selects one exact offline result. Ties are deterministic by stable id.
pub fn select_projection_autotune_record(
  records : ArrayView[ProjectionAutotuneRecord],
  device_class : String,
  numeric_format : String,
  shape : ProjectionShape,
  phase : ProjectionPhase,
  row_count : Int,
) -> ProjectionAutotuneRecord raise ProjectionStrategyError {
  if row_count <= 0 {
    raise InvalidProjectionStrategy(Bucket)
  }
  let mut selected : ProjectionAutotuneRecord? = None
  for record in records {
    if record.device_class == device_class &&
      record.numeric_format == numeric_format &&
      record.shape == shape &&
      record.strategy.phase() == phase &&
      row_count <= record.strategy.row_bucket() {
      match selected {
        None => selected = Some(record)
        Some(current) =>
          if record.median_latency_ns < current.median_latency_ns ||
            (
              record.median_latency_ns == current.median_latency_ns &&
              record.strategy.stable_id() < current.strategy.stable_id()
            ) {
            selected = Some(record)
          }
      }
    }
  }
  match selected {
    Some(record) => record
    None => raise InvalidProjectionStrategy(Measurement)
  }
}
