///|
// Original H3 releases use either one shared epsilon or three explicit
// operation epsilons. Select the dialect once during model construction;
// never silently prefer one field over a conflicting duplicate spelling.
fn original_transformer_epsilons(
  object : Map[String, Json],
) -> (Double, Double, Double) raise MiniMaxH3ConfigError {
  if object.contains("eps") {
    if ["norm_eps", "qk_norm_eps", "final_norm_eps"].any(name => {
        object.contains(name)
      }) {
      raise InvalidField
    }
    let epsilon = number_field(object, "eps")
    (epsilon, epsilon, epsilon)
  } else {
    (
      number_field(object, "norm_eps"),
      number_field(object, "qk_norm_eps"),
      number_field(object, "final_norm_eps"),
    )
  }
}
