///|
/// An offline-selected ownership envelope. It does not select arithmetic or
/// consult device measurements. Wider already-selected staging takes priority.
pub struct MatrixRowReuse {
  priv minimum_rows : Int
  priv minimum_inner : Int
  priv wide_inner : Int
  priv wide_tile : Int
} derive(Debug)

///|
pub extend MatrixRowReuse with @debug.Debug::{to_repr}

///|
pub fn MatrixRowReuse::new(
  minimum_rows~ : Int,
  minimum_inner~ : Int,
  wide_inner~ : Int,
  wide_tile~ : Int,
) -> Self raise @precision.PrecisionPlanError {
  guard minimum_rows >= 32 &&
    minimum_inner > 0 &&
    wide_inner >= minimum_inner &&
    wide_tile is (32 | 64) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  { minimum_rows, minimum_inner, wide_inner, wide_tile, }
}

///|
pub fn MatrixRowReuse::apply(
  self : Self,
  plan : MatrixTilePlan,
) -> MatrixTilePlan raise {
  let (rows, columns, inner) = plan.geometry()
  // Never undo a separately qualified wider-K or buffered transport plan.
  if rows < self.minimum_rows ||
    inner < self.minimum_inner ||
    plan.tile() != (16, 64, 32) ||
    plan.staging is Buffered(_) {
    return plan
  }
  MatrixTilePlan::new(
    rows~,
    columns~,
    inner~,
    law=plan.law(),
    tile_rows=32,
    tile_columns=32,
    staging=SynchronousTiles(
      inner_tile=if inner >= self.wide_inner { self.wide_tile } else { 32 },
    ),
  )
}
