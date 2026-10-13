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
}

///|
pub fn MatrixTilePlan::new(
  rows~ : Int,
  columns~ : Int,
  inner~ : Int,
  law~ : @precision.DenseDotLaw,
) -> Self raise @precision.PrecisionPlanError {
  guard law == MatrixFloat32 &&
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
  { rows, columns, inner, tile_rows: 16, tile_columns: 64, tile_inner: 32, }
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
  (self.rows + 15) / 16 * ((self.columns + 63) / 64)
}

///|
/// At most one nonempty padded group per job. Grouping storage consists of a
/// live group count, group expert IDs and padded original job IDs. It is
/// bounded by capacity, not by the number of experts or token values.
pub struct ExpertGroupingPlan {
  priv jobs : Int
  priv shared_input_staging : Bool
}

///|
pub fn ExpertGroupingPlan::new(
  maximum_rows~ : Int,
  selected_experts~ : Int,
  shared_input_staging? : Bool = false,
) -> Self raise @precision.PrecisionPlanError {
  let jobs = maximum_rows.to_int64() * selected_experts.to_int64()
  guard maximum_rows > 0 && selected_experts > 0 && jobs <= 1048576L else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  { jobs: jobs.to_int(), shared_input_staging, }
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
