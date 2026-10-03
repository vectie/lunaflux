///|
/// A one-token grouped-query fold maps query heads to matrix rows. It never
/// invents additional sequence tokens or shares KV across different requests.
/// Inactive instruction rows are explicitly masked. Probability quantization
/// is a named alternative law, not an exact rewrite of the SIMT F32 fold.
pub struct GroupedMatrixDecodeFold {
  priv heads : Int
  priv fold : OnlineFold
} derive(Debug, Eq)

///|
pub extend GroupedMatrixDecodeFold with Eq::{equal, not_equal}

///|
pub extend GroupedMatrixDecodeFold with @debug.Debug::{to_repr}

///|
pub fn GroupedMatrixDecodeFold::new(
  heads~ : Int,
  keys~ : Int,
  dimensions~ : Int,
) -> GroupedMatrixDecodeFold? {
  if heads <= 0 ||
    heads > 16 ||
    (keys != 32 && keys != 64) ||
    (dimensions != 64 && dimensions != 128) {
    None
  } else {
    OnlineFold::new(queries=16, keys~, dimensions~).map(fold => { heads, fold, })
  }
}

///|
pub fn GroupedMatrixDecodeFold::head_rows(
  self : GroupedMatrixDecodeFold,
) -> Int {
  self.heads
}

///|
pub fn GroupedMatrixDecodeFold::fold(
  self : GroupedMatrixDecodeFold,
) -> OnlineFold {
  self.fold
}

///|
pub fn GroupedMatrixDecodeFold::numeric_law(
  _self : GroupedMatrixDecodeFold,
) -> String {
  "grouped-head-matrix-bf16-probability-v1"
}

///|
pub fn GroupedMatrixDecodeFold::active_head(
  self : GroupedMatrixDecodeFold,
  row : Int,
) -> Bool {
  row >= 0 && row < self.heads
}
