///|
/// An offline-selected ownership envelope. It does not select arithmetic or
/// consult device measurements. An explicit fragment envelope precedes the
/// legacy row-reuse envelope; absent that selection, wider staging wins.
pub struct MatrixRowReuse {
  priv minimum_rows : Int32
  priv minimum_inner : Int32
  priv wide_inner : Int32
  priv wide_tile : Int32
  priv fragment_inner : Int32?
  priv fragment_columns : Int32
} derive(Debug)

///|
pub extend MatrixRowReuse with @debug.Debug::{to_repr}

///|
pub fn MatrixRowReuse::new(
  minimum_rows~ : Int32,
  minimum_inner~ : Int32,
  wide_inner~ : Int32,
  wide_tile~ : Int32,
  fragment_inner? : Int32,
  fragment_columns~ : Int32 = 4096,
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
