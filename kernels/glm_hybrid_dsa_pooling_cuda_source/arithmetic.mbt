///|
fn scalar(
  operation : @precision.SparseExpression,
  bindings : ArrayView[String],
) -> String {
  @arithmetic.scalar_expression(operation.expression(), bindings) catch {
    _ => abort("invalid static sparse numerical expression")
  }
}
