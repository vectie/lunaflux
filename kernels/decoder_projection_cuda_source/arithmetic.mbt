///|
fn projection_expression(
  operation : @precision.ProjectionExpression,
  bindings : ArrayView[String],
) -> String {
  @arithmetic.scalar_expression(operation.program(), bindings) catch {
    _ => abort("invalid closed projection program")
  }
}
