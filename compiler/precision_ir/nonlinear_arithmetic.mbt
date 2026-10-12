///|
/// F32 rounding of `x * scale + bias`, independent of device instructions.
pub(all) enum Float32AffineLaw {
  SeparateMultiplyAdd
  FusedMultiplyAdd
} derive(Debug, Eq)

///|
pub extend Float32AffineLaw with @debug.Debug::{to_repr}

///|
pub extend Float32AffineLaw with Eq::{equal, not_equal}

///|
/// Mathematically equivalent sigmoid expressions have different F32 rounding.
/// StableBranches uses exp(x)/(1+exp(x)) for negative x. DirectReciprocal always
/// uses 1/(1+exp(-x)); overflow of exp(-x) publishes zero, not a nonfinite value.
/// Neither contract authorizes an approximate exponential.
pub(all) enum SigmoidEvaluationLaw {
  StableBranches
  DirectReciprocal
} derive(Debug, Eq)

///|
pub extend SigmoidEvaluationLaw with @debug.Debug::{to_repr}

///|
pub extend SigmoidEvaluationLaw with Eq::{equal, not_equal}
