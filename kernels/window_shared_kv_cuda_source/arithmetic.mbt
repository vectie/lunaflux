///|
fn attention_expression(
  plan : @precision.WindowSharedKvPrecision,
  operation : @precision.AttentionExpression,
  bindings : ArrayView[String],
) -> String raise {
  @arithmetic.scalar_expression(
    plan
    .probability_law()
    .expression(operation, accumulation=plan.accumulation_law()),
    bindings,
  )
}
