///|
/// Translate the validated operand ABI into backend-neutral weight semantics
/// before elaboration. Source generators consume this immutable row view.
fn projection_bind_tile_weights(
  request : @luna_projection_tile_compiler.ProjectionTileCompilerRequest,
  parameters : ProjectionParameters,
) -> @luna_projection_tile_compiler.ProjectionTileCompilerRequest raise ProjectionAotError {
  if parameters.family != QkvProjection {
    return request
  }
  request.with_concatenated_weights([
    parameters.query_width,
    parameters.intermediate_width,
    parameters.intermediate_width,
  ]) catch {
    _ => raise Invalid(CompilerPolicy)
  }
}
