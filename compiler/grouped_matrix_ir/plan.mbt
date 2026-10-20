///|
/// Immutable logical tile geometry; backend launch topology is deliberately
/// absent. Matrix arithmetic authorization comes from precision IR.
pub struct MatrixTilePlan {
  priv rows : Int
  priv columns : Int
  priv inner : Int
  priv tile_rows : Int
  priv tile_columns : Int
  priv tile_inner : Int
  priv law : @precision.DenseDotLaw
  priv staging : MatrixStaging
}

///|
pub fn MatrixTilePlan::new(
  rows~ : Int,
  columns~ : Int,
  inner~ : Int,
  law~ : @precision.DenseDotLaw,
  staging? : MatrixStaging = Synchronous,
  tile_columns? : Int = 64,
  tile_rows? : Int = 16,
  minimum_row_reuse_inner? : Int = 0,
) -> Self raise @precision.PrecisionPlanError {
  guard law != ScalarFloat32 &&
    rows > 0 &&
    rows <= 1048576 &&
    columns > 0 &&
    inner > 0 &&
    columns <= 1073741760 &&
    inner <= 1073741760 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  // Resolve an explicitly supplied physical workset budget before lowering.
  // This is not a model-name rule or an automatic claim about hardware cache.
  guard minimum_row_reuse_inner >= 0 &&
    (
      (tile_rows == 16 && tile_columns is (16 | 32 | 64)) ||
      (tile_rows == 32 && tile_columns == 32)
    ) &&
    (tile_rows == 16 || staging is (Synchronous | SynchronousTiles(_))) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  // The supplied envelope selects between two legal physical candidates.
  // Short reductions retain the default ownership/staging, not a padded
  // row-reuse schedule whose publication overhead can exceed useful work.
  let (tile_rows, tile_columns, staging) = if tile_rows == 32 &&
    inner < minimum_row_reuse_inner {
    (16, 64, Synchronous)
  } else {
    (tile_rows, tile_columns, staging)
  }
  let staging = match staging {
    WorksetBoundedTiles(inner_tile~, maximum_weight_bytes~, maximum_tiles~) => {
      guard maximum_weight_bytes > 0L &&
        maximum_tiles > 0 &&
        inner_tile is (32 | 64 | 128) else {
        raise @precision.PrecisionPlanError::NoImplementation
      }
      let active_tiles = (rows.to_int64() + tile_rows.to_int64() - 1L) /
        tile_rows.to_int64() *
        (
          (columns.to_int64() + tile_columns.to_int64() - 1L) /
          tile_columns.to_int64()
        )
      if columns.to_int64() * inner.to_int64() * 2L <= maximum_weight_bytes &&
        active_tiles <= maximum_tiles.to_int64() &&
        inner % inner_tile == 0 {
        SynchronousTiles(inner_tile~)
      } else {
        Synchronous
      }
    }
    other => other
  }
  guard (tile_rows == 16 && tile_columns == 64) ||
    staging is (Synchronous | SynchronousTiles(_)) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  guard (rows.to_int64() + tile_rows.to_int64() - 1L) /
    tile_rows.to_int64() *
    (
      (columns.to_int64() + tile_columns.to_int64() - 1L) /
      tile_columns.to_int64()
    ) <=
    2147483647L else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let tile_inner = match staging {
    Synchronous => 32
    SynchronousTiles(inner_tile~) | Buffered(inner_tile~) => {
      guard inner_tile is (32 | 64 | 128) else {
        raise @precision.PrecisionPlanError::NoImplementation
      }
      inner_tile
    }
    WorksetBoundedTiles(..) =>
      raise @precision.PrecisionPlanError::NoImplementation
  }
  { rows, columns, inner, tile_rows, tile_columns, tile_inner, law, staging, }
}

///|
pub fn MatrixTilePlan::geometry(self : Self) -> (Int, Int, Int) {
  (self.rows, self.columns, self.inner)
}

///|
pub fn MatrixTilePlan::tile(self : Self) -> (Int, Int, Int) {
  (self.tile_rows, self.tile_columns, self.tile_inner)
}

///|
pub fn MatrixTilePlan::tile_count(self : Self) -> Int {
  let matrix = (self.rows + self.tile_rows - 1) /
    self.tile_rows *
    ((self.columns + self.tile_columns - 1) / self.tile_columns)
  if self.law == ShapeAwareFloat32 {
    Int::max(matrix, (self.columns + 3) / 4)
  } else {
    matrix
  }
}

///|
pub fn MatrixTilePlan::law(self : Self) -> @precision.DenseDotLaw {
  self.law
}

///|
/// At most one nonempty padded group per job. Grouping storage consists of a
/// live group count, group expert IDs and padded original job IDs. It is
/// bounded by capacity, not by the number of experts or token values.
pub struct ExpertGroupingPlan {
  priv jobs : Int
  priv shared_input_staging : Bool
  priv column_tile : Int
  priv transport : ExpertTransport
  priv fragment_slots : Int
  priv matrix_columns : Int
  priv scale_partitions : Int
}

///|
pub(all) enum ExpertTransport {
  IndexedRows
  IndexedBuffered
  PackedRows
  PackedPipeline
  PackedBulk(columns~ : Int, consumer_teams~ : Int, workers~ : Int)
  PackedBulkTiles(columns~ : Int, consumer_teams~ : Int, workers~ : Int)
} derive(Eq, Debug)

///|
pub extend ExpertTransport with Eq::{equal, not_equal}

///|
pub extend ExpertTransport with @debug.Debug::{to_repr}

///|
pub fn ExpertGroupingPlan::new(
  maximum_rows~ : Int,
  selected_experts~ : Int,
  shared_input_staging? : Bool = false,
  column_tile? : Int = 32,
  transport? : ExpertTransport = IndexedRows,
  fragment_slots? : Int = 1,
  matrix_columns? : Int = 64,
  scale_partitions? : Int = 1,
) -> Self raise @precision.PrecisionPlanError {
  guard scale_partitions > 0 &&
    scale_partitions <= 256 &&
    fragment_slots is (1 | 2) &&
    matrix_columns is (16 | 32 | 64) &&
    (matrix_columns == 64 || transport == IndexedRows) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let column_tile = match transport {
    PackedBulk(columns~, consumer_teams~, workers~)
    | PackedBulkTiles(columns~, consumer_teams~, workers~) => {
      guard columns is (64 | 128) &&
        consumer_teams is (1 | 2) &&
        workers > 0 &&
        workers <= 1024 else {
        raise @precision.PrecisionPlanError::NoImplementation
      }
      columns
    }
    _ => column_tile
  }
  let jobs = maximum_rows.to_int64() * selected_experts.to_int64()
  guard maximum_rows > 0 &&
    selected_experts > 0 &&
    jobs <= 1048576L &&
    column_tile > 0 &&
    column_tile <= 1024 else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    jobs: jobs.to_int(),
    shared_input_staging,
    column_tile,
    transport,
    fragment_slots,
    matrix_columns,
    scale_partitions,
  }
}

///|
/// Physical maximum producers; independent of expert grouping and arithmetic.
pub fn ExpertGroupingPlan::scale_partitions(self : Self) -> Int {
  self.scale_partitions
}

///|
/// Independent output-column ownership for the indexed BF16 matrix schedule.
/// It changes parallel work distribution, not any dot-product reduction order.
pub fn ExpertGroupingPlan::matrix_columns(self : Self) -> Int {
  self.matrix_columns
}

///|
/// Read-ahead ownership inside a published operand epoch, independent of the
/// global-to-local transport and the accumulator's ascending numerical fold.
pub fn ExpertGroupingPlan::fragment_slots(self : Self) -> Int {
  self.fragment_slots
}

///|
pub fn ExpertGroupingPlan::transport(self : Self) -> ExpertTransport {
  self.transport
}

///|
pub fn ExpertGroupingPlan::maximum_groups(self : Self) -> Int {
  self.jobs
}

///|
pub fn ExpertGroupingPlan::rows_per_group(_self : Self) -> Int {
  16
}

///|
pub fn ExpertGroupingPlan::storage_bytes(self : Self) -> Int64 {
  (1L + self.jobs.to_int64() * 17L) * 4L
}

///|
/// Physical reuse, independent of numerical reduction/publication laws. The
/// measured frontier may retain both shared and independent staging schedules.
pub fn ExpertGroupingPlan::shared_input_staging(self : Self) -> Bool {
  self.shared_input_staging
}

///|
/// Output ownership width. Independent output columns may reuse an operand
/// without changing any column's ordered reduction or numerical law.
pub fn ExpertGroupingPlan::column_tile(self : Self) -> Int {
  self.column_tile
}

///|
/// Physical independent consumer domains; arithmetic is unchanged.
pub fn ExpertGroupingPlan::consumer_teams(self : Self) -> Int {
  match self.transport {
    PackedBulk(consumer_teams~, ..) | PackedBulkTiles(consumer_teams~, ..) =>
      consumer_teams
    _ => 1
  }
}

///|
/// A finite persistent worker pool partitions the live indirect tile domain.
pub fn ExpertGroupingPlan::persistent_workers(self : Self) -> Int? {
  match self.transport {
    PackedBulk(workers~, ..) | PackedBulkTiles(workers~, ..) => Some(workers)
    _ => None
  }
}
