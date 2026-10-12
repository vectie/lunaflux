///|
/// Offline measurements for one complete plan, not a sum of kernel speedups.
/// Callers supply only plans with the same logical/state contract and already
/// evaluated against their fixed external quality requirements. That filtering
/// is NOT a compiler proof. Ordered folds retain seed/extent and update order.
pub(all) struct MeasuredNumericalPlan {
  label : String
  expressions : ReadOnlyArray[@precision.NumericExpression]
  folds : ReadOnlyArray[@precision.OrderedNumericFold]
  elapsed_ns : Int64
  memory_bytes : Int64
}

///|
fn exact_expression(
  a : @precision.NumericExpression,
  b : @precision.NumericExpression,
) -> Bool {
  match a.relation_to(b) {
    IdenticalProgram | ExactFiniteBf16Roundtrip => true
    _ => false
  }
}

///|
/// Additional numerical filter for an EXACT-program experiment. Even perfect
/// output agreement cannot admit an unproven rewrite. A different declared
/// program is a separate experiment, not an optimizer substitution. This API
/// does not certify the caller's state/layout mapping or whole-model quality.
/// The compiler never imports this package or consumes elapsed/quality scores.
pub fn select_exact_program(
  expressions : ArrayView[@precision.NumericExpression],
  folds : ArrayView[@precision.OrderedNumericFold],
  candidates : ArrayView[MeasuredNumericalPlan],
  memory_ceiling~ : Int64,
) -> Int? {
  for index in 0..<candidates.length(); best = (None : Int?) {
    let item = candidates[index]
    if item.elapsed_ns <= 0L ||
      item.memory_bytes < 0L ||
      item.memory_bytes > memory_ceiling ||
      item.expressions.length() != expressions.length() ||
      item.folds.length() != folds.length() {
      continue best
    }
    let expressions_match = expressions
      .mapi((i, expression) => exact_expression(expression, item.expressions[i]))
      .all(value => value)
    let folds_match = folds
      .mapi((i, fold) => {
        let candidate = item.folds[i]
        candidate.extent == fold.extent &&
        candidate.initial_bits == fold.initial_bits &&
        exact_expression(fold.update, candidate.update)
      })
      .all(value => value)
    if !expressions_match || !folds_match {
      continue best
    }
    continue match best {
        Some(previous) if candidates[previous].elapsed_ns <= item.elapsed_ns =>
          best
        _ => Some(index)
      }
  } nobreak {
    best
  }
}
