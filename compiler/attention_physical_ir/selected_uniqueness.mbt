///|
/// Exact membership validation, independent of attention's numeric reductions.
/// A bounded bitmap is reused only after every worker finishes its window.
pub enum SelectedUniquenessSchedule {
  Pairwise
  WindowedBitmap(words~ : Int)
} derive(Eq)

///|
pub extend SelectedUniquenessSchedule with Eq::{equal, not_equal}

///|
pub fn SelectedSparseSchedule::uniqueness(
  self : Self,
  extent : Int,
) -> SelectedUniquenessSchedule? {
  guard extent >= 0 && extent <= self.selected_width else { return None }
  let words = Int::min(
    self.workers,
    ((self.history.to_int64() + 31L) / 32L).to_int(),
  )
  let window = words.to_int64() * 32L
  let passes = (self.history.to_int64() + window - 1L) / window
  let pairwise = extent.to_int64() * (extent - 1).to_int64() / 2L
  if passes * (extent.to_int64() + words.to_int64()) < pairwise {
    Some(WindowedBitmap(words~))
  } else {
    Some(Pairwise)
  }
}
