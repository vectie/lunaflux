///|
/// An offline-selected ownership envelope. It does not select arithmetic or
/// consult device measurements. An explicit fragment envelope precedes the
/// legacy row-reuse envelope; absent that selection, wider staging wins.
pub struct MatrixRowReuse {
  priv minimum_rows : Int
  priv minimum_inner : Int
  priv wide_inner : Int
  priv wide_tile : Int
  priv fragment_inner : Int?
  priv fragment_columns : Int
} derive(Debug)

///|
pub extend MatrixRowReuse with @debug.Debug::{to_repr}

///|
pub fn MatrixRowReuse::new(
  minimum_rows~ : Int,
  minimum_inner~ : Int,
  wide_inner~ : Int,
  wide_tile~ : Int,
  fragment_inner? : Int,
  fragment_columns~ : Int = 4096,
) -> Self raise @precision.PrecisionPlanError {
  guard minimum_rows >= 32 &&
    minimum_inner > 0 &&
    wide_inner >= minimum_inner &&
    wide_tile is (32 | 64) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  guard fragment_columns > 0 &&
    (fragment_inner is None || fragment_inner.unwrap() > 0) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  {
    minimum_rows,
    minimum_inner,
    wide_inner,
    wide_tile,
    fragment_inner,
    fragment_columns,
  }
}

///|
pub fn MatrixRowReuse::apply(
  self : Self,
  plan : MatrixTilePlan,
) -> MatrixTilePlan raise {
  let (rows, columns, inner) = plan.geometry()
  if self.fragment_inner is Some(minimum) &&
    rows >= self.minimum_rows &&
    inner >= minimum &&
    inner % 64 == 0 &&
    columns >= self.fragment_columns {
    return MatrixTilePlan::new(
      rows~,
      columns~,
      inner~,
      law=plan.law(),
      staging=FragmentPipeline(row_tile=32),
    )
  }
  // Never undo a separately qualified wider-K or buffered transport plan.
  if rows < self.minimum_rows ||
    inner < self.minimum_inner ||
    plan.tile() != (16, 64, 32) ||
    plan.staging is (Buffered(_) | FragmentPipeline(_)) {
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
