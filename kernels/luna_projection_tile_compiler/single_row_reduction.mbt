///|
/// Reassociation is an explicit numerical permission, not a consequence of
/// purity or CSE. The caller must validate the resulting rounding differences.
pub(all) enum ProjectionSingleRowNumerics {
  PreserveDotOrder
  AllowDeterministicDotTree
} derive(Debug, Eq)

///|
/// A pure schedule for a dot product. StridedPairwiseDot folds products at
/// lane + n * lanes in increasing n, then combines lane partials with a
/// descending power-of-two tree. Products and additions round separately.
/// It is deterministic for a fixed topology, not bit-exact to OrderedDot.
/// Subgroup instruction names and lane-movement primitives belong in lowering.
pub(all) enum ProjectionSingleRowReduction {
  OrderedDot
  StridedPairwiseDot(Int)
} derive(Debug, Eq)

///|
fn projection_single_row_reduction(
  request : ProjectionTileCompilerRequest,
) -> ProjectionSingleRowReduction {
  let decode = request.dispatch.decode()
  let (_, _, lanes) = decode.tile().dimensions()
  if request.single_row_numerics == AllowDeterministicDotTree &&
    decode.kernel_class() == SubgroupGemv &&
    lanes > 1 &&
    (lanes & (lanes - 1)) == 0 {
    StridedPairwiseDot(lanes)
  } else {
    OrderedDot
  }
}

///|
pub fn ProjectionTileSchedule::single_row_reduction(
  self : ProjectionTileSchedule,
) -> ProjectionSingleRowReduction {
  self.single_row_reduction
}
