///|
/// Operand and operator-publication precision under an explicit autocast
/// region. Accumulation, normalization and residual precision remain separate
/// contracts. These equal-width storage formats are not interchangeable.
pub(all) enum AutocastFloat {
  Float16
  Bfloat16
} derive(Debug, Eq)

///|
pub extend AutocastFloat with @debug.Debug::{to_repr}

///|
pub extend AutocastFloat with Eq::{equal, not_equal}
