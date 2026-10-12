///|
fn weight_expression(
  operation : @precision.RoutingWeightExpression,
  bindings : ArrayView[String],
) -> String {
  @arithmetic.scalar_expression(operation.program(), bindings) catch {
    _ => abort("invalid static routing expression")
  }
}
