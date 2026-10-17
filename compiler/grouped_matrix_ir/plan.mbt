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
  guard (rows.to_int64() + 15L) / 16L * ((columns.to_int64() + 63L) / 64L) <=
    2147483647L else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let tile_inner = match staging {
    Synchronous => 32
    Buffered(inner_tile~) => {
      guard inner_tile is (32 | 64 | 128) else {
        raise @precision.PrecisionPlanError::NoImplementation
      }
      inner_tile
    }
  }
  {
    rows,
    columns,
    inner,
    tile_rows: 16,
    tile_columns: 64,
    tile_inner,
    law,
    staging,
  }
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
  let matrix = (self.rows + 15) / 16 * ((self.columns + 63) / 64)
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
}

///|
pub(all) enum ExpertTransport {
  IndexedRows
  IndexedBuffered
  PackedRows
  PackedPipeline
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
) -> Self raise @precision.PrecisionPlanError {
  guard fragment_slots is (1 | 2) else {
    raise @precision.PrecisionPlanError::NoImplementation
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
  }
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
