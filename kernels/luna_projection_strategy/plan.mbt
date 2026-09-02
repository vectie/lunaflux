///|
const PROJECTION_BUCKET_SLOT_COUNT : Int = 19

///|
fn projection_row_bucket_slot(
  row_count : Int,
) -> Int raise ProjectionStrategyError {
  if row_count <= 0 || row_count > 1048576 {
    raise InvalidProjectionStrategy(Bucket)
  }
  if row_count == 1 {
    0
  } else if row_count <= 8 {
    1
  } else if row_count <= 16 {
    2
  } else if row_count <= 32 {
    3
  } else if row_count <= 64 {
    4
  } else if row_count <= 128 {
    5
  } else if row_count <= 256 {
    6
  } else if row_count <= 512 {
    7
  } else if row_count <= 1024 {
    8
  } else if row_count <= 2048 {
    9
  } else if row_count <= 4096 {
    10
  } else if row_count <= 8192 {
    11
  } else if row_count <= 16384 {
    12
  } else if row_count <= 32768 {
    13
  } else if row_count <= 65536 {
    14
  } else if row_count <= 131072 {
    15
  } else if row_count <= 262144 {
    16
  } else if row_count <= 524288 {
    17
  } else {
    18
  }
}

///|
fn projection_bucket_for_slot(slot : Int) -> Int raise ProjectionStrategyError {
  match slot {
    0 => 1
    1 => 8
    2 => 16
    3 => 32
    4 => 64
    5 => 128
    6 => 256
    7 => 512
    8 => 1024
    9 => 2048
    10 => 4096
    11 => 8192
    12 => 16384
    13 => 32768
    14 => 65536
    15 => 131072
    16 => 262144
    17 => 524288
    18 => 1048576
    _ => raise InvalidProjectionStrategy(Bucket)
  }
}

///|
pub fn projection_row_bucket(
  maximum_rows : Int,
) -> Int raise ProjectionStrategyError {
  projection_bucket_for_slot(projection_row_bucket_slot(maximum_rows))
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
fn static_projection_strategy(
  shape : ProjectionShape,
  row_bucket : Int,
  capabilities : ProjectionCapabilities,
) -> ProjectionStrategy raise ProjectionStrategyError {
  let subgroup = capabilities.subgroup_width()
  if row_bucket == 1 {
    // Gated MLP decode can use the subgroup path only when the backend also
    // supplies its matrix-tiled gate/up/down companion. Without that
    // companion the admitted implementation is the scalar fused kernel for
    // every row count, so the strategy must not claim an unavailable GEMV.
    let gated_mlp_subgroup_supported = match capabilities.matrix_tile() {
      Some(tile) => projection_matrix_eligible(shape, tile)
      None => false
    }
    let kernel_class = if shape.input_width % subgroup == 0 &&
      (shape.family() != GatedMlp || gated_mlp_subgroup_supported) {
      SubgroupGemv
    } else {
      ScalarTile
    }
    ProjectionStrategy::new(
      stable_id=if kernel_class == SubgroupGemv { 101 } else { 100 },
      phase=Decode,
      strategy_class=Some(DecodeGemv),
      kernel_class~,
      row_bucket=1,
      tile=ProjectionTile::new(rows=1, columns=subgroup, reduction=subgroup),
    )
  } else {
    let strategy_class = if row_bucket <= 8 {
      SmallRowTiledGemm
    } else {
      LargeRowTiledGemm
    }
    let (kernel_class, tile, stable_id) = match capabilities.matrix_tile() {
      Some(tile) if projection_matrix_eligible(shape, tile) =>
        (MatrixTile, tile, 201)
      _ =>
        (
          ScalarTile,
          ProjectionTile::new(rows=1, columns=subgroup, reduction=1),
          200,
        )
    }
    ProjectionStrategy::new(
      stable_id~,
      phase=Prefill,
      strategy_class=Some(strategy_class),
      kernel_class~,
      row_bucket~,
      tile~,
    )
  }
}

///|
pub fn plan_projection_dispatch(
  shape : ProjectionShape,
  maximum_prefill_rows : Int,
  capabilities : ProjectionCapabilities,
) -> ProjectionDispatchPlan raise ProjectionStrategyError {
  let row_bucket = projection_row_bucket(maximum_prefill_rows)
  let decode = static_projection_strategy(shape, 1, capabilities)
  let prefill = static_projection_strategy(shape, row_bucket, capabilities)
  { shape, decode, prefill }
}

///|
fn compatible_projection_candidate(
  shape : ProjectionShape,
  row_bucket : Int,
  strategy : ProjectionStrategy,
  capabilities : ProjectionCapabilities,
) -> Bool {
  if strategy.row_bucket() != row_bucket {
    return false
  }
  let expected_phase = if row_bucket == 1 { Decode } else { Prefill }
  let expected_class = if row_bucket == 1 {
    DecodeGemv
  } else if row_bucket <= 8 {
    SmallRowTiledGemm
  } else {
    LargeRowTiledGemm
  }
  if strategy.phase() != expected_phase ||
    strategy.strategy_class() != expected_class {
    return false
  }
  match strategy.kernel_class() {
    ScalarTile => true
    SubgroupGemv =>
      row_bucket == 1 && shape.input_width % capabilities.subgroup_width() == 0
    MatrixTile =>
      row_bucket > 1 && projection_matrix_eligible(shape, strategy.tile())
  }
}

///|
fn validate_projection_autotune_records(
  records : ArrayView[ProjectionAutotuneRecord],
) -> Unit raise ProjectionStrategyError {
  for left_index, left in records {
    let mut right_index = left_index + 1
    while right_index < records.length() {
      let right = records[right_index]
      if left.key() == right.key() {
        if left.median_latency_ns() == right.median_latency_ns() &&
          left.sample_count() == right.sample_count() {
          raise InvalidProjectionStrategy(Duplicate)
        }
        raise InvalidProjectionStrategy(Conflict)
      }
      let left_key = left.key()
      let right_key = right.key()
      let left_strategy = left_key.strategy()
      let right_strategy = right_key.strategy()
      if left_key.backend() == right_key.backend() &&
        left_key.target() == right_key.target() &&
        left_key.numeric_type() == right_key.numeric_type() &&
        left_key.shape() == right_key.shape() &&
        left_strategy.phase() == right_strategy.phase() &&
        left_strategy.row_bucket() == right_strategy.row_bucket() &&
        left_strategy.stable_id() == right_strategy.stable_id() {
        raise InvalidProjectionStrategy(Conflict)
      }
      right_index += 1
    }
  }
}

///|
fn better_projection_record(
  candidate : ProjectionAutotuneRecord,
  current : ProjectionAutotuneRecord,
) -> Bool {
  candidate.median_latency_ns() < current.median_latency_ns() ||
  (
    candidate.median_latency_ns() == current.median_latency_ns() &&
    candidate.strategy().stable_id() < current.strategy().stable_id()
  )
}

///|
fn select_projection_record_for_bucket(
  records : ArrayView[ProjectionAutotuneRecord],
  backend : ProjectionBackendId,
  target : ProjectionTargetId,
  numeric_type : ProjectionNumericType,
  shape : ProjectionShape,
  row_bucket : Int,
  capabilities : ProjectionCapabilities,
) -> ProjectionAutotuneRecord? {
  let mut selected : ProjectionAutotuneRecord? = None
  for record in records {
    let key = record.key()
    if key.backend() == backend &&
      key.target() == target &&
      key.numeric_type() == numeric_type &&
      key.shape() == shape &&
      compatible_projection_candidate(
        shape,
        row_bucket,
        key.strategy(),
        capabilities,
      ) {
      match selected {
        None => selected = Some(record)
        Some(current) =>
          if better_projection_record(record, current) {
            selected = Some(record)
          }
      }
    }
  }
  selected
}

///|
/// Builds the bounded runtime table from immutable offline records. Validation,
/// candidate scans, and allocation happen here, never in token-step dispatch.
pub fn build_projection_runtime_dispatch(
  records : ArrayView[ProjectionAutotuneRecord],
  backend~ : ProjectionBackendId,
  target~ : ProjectionTargetId,
  numeric_type~ : ProjectionNumericType,
  shape~ : ProjectionShape,
  maximum_rows~ : Int,
  capabilities~ : ProjectionCapabilities,
) -> ProjectionRuntimeDispatch raise ProjectionStrategyError {
  validate_projection_autotune_records(records)
  let maximum_slot = projection_row_bucket_slot(maximum_rows)
  let initial = ProjectionSelectedStrategy::{
    strategy: static_projection_strategy(shape, 1, capabilities),
    source: StaticFallback,
  }
  let slots = FixedArray::make(PROJECTION_BUCKET_SLOT_COUNT, initial)
  let mut slot = 0
  while slot <= maximum_slot {
    let row_bucket = projection_bucket_for_slot(slot)
    let fallback = static_projection_strategy(shape, row_bucket, capabilities)
    slots[slot] = match
      select_projection_record_for_bucket(
        records, backend, target, numeric_type, shape, row_bucket, capabilities,
      ) {
      Some(record) => { strategy: record.strategy(), source: OfflineAutotune }
      None => { strategy: fallback, source: StaticFallback }
    }
    slot += 1
  }
  { backend, target, numeric_type, shape, maximum_rows, slots }
}

///|
/// Constant-time, allocation-free token-step selection.
pub fn ProjectionRuntimeDispatch::select(
  self : ProjectionRuntimeDispatch,
  row_count : Int,
) -> ProjectionSelectedStrategy raise ProjectionStrategyError {
  if row_count <= 0 || row_count > self.maximum_rows {
    raise InvalidProjectionStrategy(Bucket)
  }
  self.slots[projection_row_bucket_slot(row_count)]
}

///|
/// Freezes the decode and admitted prefill selections for one AOT lowering.
/// This is a startup operation; request-time code should retain the runtime
/// table and call `select` directly.
pub fn ProjectionRuntimeDispatch::dispatch_plan(
  self : ProjectionRuntimeDispatch,
  maximum_prefill_rows : Int,
) -> ProjectionDispatchPlan raise ProjectionStrategyError {
  let decode = self.select(1).strategy()
  let prefill = self.select(maximum_prefill_rows).strategy()
  { shape: self.shape, decode, prefill }
}
