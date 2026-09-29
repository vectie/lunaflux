///|
/// Explicit alternative numerical law: QK dots retain their subgroup tree,
/// tile probabilities stay F32, and one max/denominator/numerator state is
/// merged per ascending key tile. No BF16 probability rounding is permitted.
pub struct BlockwiseFold {
  priv dimensions : Int
  priv keys : Int
} derive(Debug, Eq)

///|
pub extend BlockwiseFold with Eq::{equal, not_equal}

///|
pub extend BlockwiseFold with @debug.Debug::{to_repr}

///|
pub fn BlockwiseFold::new(dimensions~ : Int, keys~ : Int) -> BlockwiseFold? {
  if dimensions > 0 && keys > 0 {
    Some({ dimensions, keys, })
  } else {
    None
  }
}

///|
pub fn BlockwiseFold::dimensions(self : BlockwiseFold) -> (Int, Int) {
  (self.dimensions, self.keys)
}

///|
pub fn BlockwiseFold::numeric_law(_self : BlockwiseFold) -> String {
  "blockwise-f32-probability-v1"
}

///|
/// Pure state/effect ordering. The tile-local score exchange is subgroup
/// scoped; shared K/V lifetime remains the enclosing workgroup effect plan.
pub(all) enum BlockwiseFoldStep {
  ComputeTileScores
  PublishTileScores
  ReduceTileMaximum
  FormFloat32Probabilities
  ReduceTileDenominator
  PublishTileProbabilities
  AccumulateTileValues
  MergeTileState
} derive(Debug, Eq)

///|
pub extend BlockwiseFoldStep with Eq::{equal, not_equal}

///|
pub extend BlockwiseFoldStep with @debug.Debug::{to_repr}

///|
pub fn BlockwiseFold::steps(
  _self : BlockwiseFold,
) -> ReadOnlyArray[BlockwiseFoldStep] {
  [
    ComputeTileScores,
    PublishTileScores,
    ReduceTileMaximum,
    FormFloat32Probabilities,
    ReduceTileDenominator,
    PublishTileProbabilities,
    AccumulateTileValues,
    MergeTileState,
  ]
}
