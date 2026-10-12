///|
fn control_expression(
  operation : @precision.HyperControlExpression,
  bindings : ArrayView[String],
) -> String {
  @arithmetic.scalar_expression(operation.expression(), bindings) catch {
    _ => abort("invalid static control expression")
  }
}
