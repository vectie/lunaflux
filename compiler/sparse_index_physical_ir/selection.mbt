///|
/// Two exact physical decompositions of stable top-k. Neither changes score
/// arithmetic or tie order. Independent ranks consume immutable scores;
/// cooperative maxima publish one winner before the next selection round.
pub enum StableSelectionSchedule {
  IndependentRanks
  CooperativeMaxima
} derive(Eq)

///|
pub extend StableSelectionSchedule with Eq::{equal, not_equal}

///|
/// Bound pairwise ranking work by the existing sequential selection work per
/// worker. Large domains use cooperative maxima instead of quadratic ranking.
/// This is an analytic work bound, not a measured performance claim.
pub fn RetainedIndexPlan::selection(
  self : Self,
  workers : Int,
) -> StableSelectionSchedule? {
  guard workers > 0 else { return None }
  if self.candidates.to_int64() <=
    self.selected_pools.to_int64() * workers.to_int64() {
    Some(IndependentRanks)
  } else {
    Some(CooperativeMaxima)
  }
}
