///|
/// Explicit alternative numerical law: QK dots use declared owner-local trees,
/// tile probabilities stay F32, and one max/denominator/numerator state is
/// merged per ascending key tile. No BF16 probability rounding is permitted.
pub struct BlockwiseFold {
  priv dimensions : Int
  priv keys : Int
  priv score_owners : Int
  priv contraction : Bool
} derive(Debug, Eq)

///|
pub extend BlockwiseFold with Eq::{equal, not_equal}

///|
pub extend BlockwiseFold with @debug.Debug::{to_repr}

///|
pub fn BlockwiseFold::new(
  dimensions~ : Int,
  keys~ : Int,
  score_owners? : Int = 1,
  contraction? : Bool = false,
) -> BlockwiseFold? {
  if dimensions > 0 &&
    keys > 0 &&
    (score_owners == 1 || score_owners == 2) &&
    (!contraction || score_owners == 1) {
    Some({ dimensions, keys, score_owners, contraction, })
  } else {
    None
  }
}

///|
pub fn BlockwiseFold::dimensions(self : BlockwiseFold) -> (Int, Int) {
  (self.dimensions, self.keys)
}

///|
pub fn BlockwiseFold::numeric_law(self : BlockwiseFold) -> String {
  if self.contraction {
    "blockwise-fma-f32-probability-v3"
  } else if self.score_owners == 1 {
    "blockwise-f32-probability-v1"
  } else {
    "dual-score-blockwise-f32-probability-v2"
  }
}

///|
pub fn BlockwiseFold::contracts_multiply_add(self : BlockwiseFold) -> Bool {
  self.contraction
}

///|
/// Number of independent score reductions sharing a physical subgroup.
/// It is a portable ownership choice, not a device instruction or lane count.
pub fn BlockwiseFold::score_owners(self : BlockwiseFold) -> Int {
  self.score_owners
}

///|
/// Independent owners traverse a cyclic permutation of component groups.
/// This is a bijection: all components occur once, including a ragged tail.
/// A device backend may exploit the different traversal phases to spread
/// shared-memory accesses without introducing another staging buffer.
pub fn BlockwiseFold::rotates_score_groups(self : BlockwiseFold) -> Bool {
  self.score_owners > 1
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
