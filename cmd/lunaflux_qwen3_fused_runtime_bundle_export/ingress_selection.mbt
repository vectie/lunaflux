///|
/// A compatibility default, a requested experiment and an offline measured
/// winner are different decisions even when they name the same fusion cut.
priv enum IngressSelectionBasis {
  CompatibilityDefault
  ExplicitEvaluation
  MeasuredComparison
}

///|
fn IngressSelectionBasis::canonical_name(
  self : IngressSelectionBasis,
) -> String {
  match self {
    CompatibilityDefault => "compatibility-default"
    ExplicitEvaluation => "explicit-evaluation"
    MeasuredComparison => "measured-comparison"
  }
}
