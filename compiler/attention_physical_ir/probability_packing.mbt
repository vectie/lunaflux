///|
/// An immutable streaming consumer of ordered probability elements. A packed
/// word is published immediately after its elements contribute to their row's
/// denominator. Only one word's scalar inputs are live: no whole-fragment
/// exponential array or deferred permutation is necessary.
pub struct OrderedProbabilityPacking {
  priv program : OnlineFragmentProgram
  priv owners : ReadOnlyArray[Int]
  priv elements_per_word : Int
} derive(Eq, Debug)

///|
pub extend OrderedProbabilityPacking with Eq::{equal, not_equal}

///|
pub extend OrderedProbabilityPacking with @debug.Debug::{to_repr}

///|
pub fn OnlineFragmentProgram::pack_probabilities(
  self : OnlineFragmentProgram,
  element_owners : ArrayView[Int],
  elements_per_word~ : Int,
) -> OrderedProbabilityPacking? {
  guard elements_per_word > 0 &&
    elements_per_word <= 64 &&
    element_owners.length() > 0 &&
    element_owners.length() <= 1024 &&
    element_owners.length() % elements_per_word == 0 else {
    return None
  }
  for owner in element_owners {
    guard owner >= 0 && owner < self.geometry().0 else { return None }
  }
  Some({
    program: self,
    owners: ReadOnlyArray::from_array(element_owners),
    elements_per_word,
  })
}

///|
pub fn OrderedProbabilityPacking::program(
  self : OrderedProbabilityPacking,
) -> OnlineFragmentProgram {
  self.program
}

///|
pub fn OrderedProbabilityPacking::geometry(
  self : OrderedProbabilityPacking,
) -> (Int, Int) {
  (self.owners.length() / self.elements_per_word, self.elements_per_word)
}

///|
pub fn OrderedProbabilityPacking::owner(
  self : OrderedProbabilityPacking,
  word : Int,
  element : Int,
) -> Int? {
  guard word >= 0 &&
    word < self.geometry().0 &&
    element >= 0 &&
    element < self.elements_per_word else {
    return None
  }
  Some(self.owners[word * self.elements_per_word + element])
}

///|
pub fn OrderedProbabilityPacking::peak_scalar_inputs(
  self : OrderedProbabilityPacking,
) -> Int {
  self.elements_per_word
}
