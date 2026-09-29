///|
/// Matrix ownership for a query-owned online fold. Q and rounded probabilities
/// are reused across right columns; each K/V fragment has one product consumer.
/// Geometry is supplied by the target's instruction contract, not inferred
/// from a model name. The outer ascending-key fold and its effects are unchanged.
pub struct OnlineFragmentProgram {
  priv fold : OnlineFold
  priv rows : Int
  priv columns : Int
  priv quantum : Int
  priv score : @physical_tile_ir.FragmentProgram
  priv value : @physical_tile_ir.FragmentProgram
} derive(Eq, Debug)

///|
pub extend OnlineFragmentProgram with Eq::{equal, not_equal}

///|
pub extend OnlineFragmentProgram with @debug.Debug::{to_repr}

///|
pub fn OnlineFold::fragment_products(
  self : OnlineFold,
  fragment_rows~ : Int,
  fragment_columns~ : Int,
  reduction_quantum~ : Int,
) -> OnlineFragmentProgram? {
  let (queries, keys, dimensions) = self.geometry()
  guard fragment_rows > 0 &&
    fragment_columns > 0 &&
    reduction_quantum > 0 &&
    queries % fragment_rows == 0 &&
    keys % fragment_columns == 0 &&
    dimensions % fragment_columns == 0 else {
    return None
  }
  guard @physical_tile_ir.FragmentProgram::new(
      rows=1,
      columns=keys / fragment_columns,
      extent=dimensions,
      quantum=reduction_quantum,
      slots=1,
    )
    is Some(score) else {
    return None
  }
  guard @physical_tile_ir.FragmentProgram::new(
      rows=1,
      columns=dimensions / fragment_columns,
      extent=keys,
      quantum=reduction_quantum,
      slots=1,
    )
    is Some(value) else {
    return None
  }
  Some({
    fold: self,
    rows: fragment_rows,
    columns: fragment_columns,
    quantum: reduction_quantum,
    score,
    value,
  })
}

///|
pub fn OnlineFragmentProgram::fold(self : OnlineFragmentProgram) -> OnlineFold {
  self.fold
}

///|
pub fn OnlineFragmentProgram::geometry(
  self : OnlineFragmentProgram,
) -> (Int, Int, Int) {
  (self.rows, self.columns, self.quantum)
}

///|
pub fn OnlineFragmentProgram::score(
  self : OnlineFragmentProgram,
) -> @physical_tile_ir.FragmentProgram {
  self.score
}

///|
pub fn OnlineFragmentProgram::value(
  self : OnlineFragmentProgram,
) -> @physical_tile_ir.FragmentProgram {
  self.value
}
