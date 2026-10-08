///|
/// Semantic output ownership in a mixed attention step. A complete writer
/// consumes original CSR rows; a prefill-only writer needs a disjoint decode
/// companion. Neither choice changes the pure-decode implementation.
pub(all) enum AttentionMixedRowCoverage {
  PrefillRowsOnly
  AllActiveRows
} derive(Debug, Eq)
