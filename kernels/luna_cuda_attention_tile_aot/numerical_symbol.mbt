///|
/// Actual physical arithmetic, not frontier permission, owns the namespace.
fn cuda_attention_numerical_symbol(
  requested : String,
  compute : @luna_attention_strategy.AttentionTileCompute,
) -> String {
  if compute == AttentionMatrixQkPvApproxExp {
    if requested.contains("_exp2_v1") {
      requested
    } else {
      requested + "_exp2_v1"
    }
  } else {
    requested.replace(old="_exp2_v1", new="_v1")
  }
}
