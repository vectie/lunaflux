///|
fn rms_expression(
  operation : @precision.RmsExpression,
  bindings : ArrayView[String],
) -> String {
  @arithmetic.scalar_expression(operation.program(), bindings) catch {
    _ => abort("invalid closed RMS program")
  }
}

///|
fn rms_inverse(sum : String, width : Int, epsilon : String) -> String {
  let variance = rms_expression(Variance, [sum, "\{width}.0f", epsilon])
  @arithmetic.scalar_expression(
    @precision.ReciprocalRootLaw::ReciprocalSqrtFloat32.expression(),
    [variance],
  ) catch {
    _ => abort("invalid RMS root program")
  }
}
