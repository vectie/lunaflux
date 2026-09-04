///|
pub fn ProjectionTileCompilerRequest::new(
  dispatch~ : @luna_projection_strategy.ProjectionDispatchPlan,
  maximum_rows~ : Int,
  maximum_tokens~ : Int,
  demand~ : ProjectionResultDemand,
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  if maximum_rows <= 0 ||
    maximum_tokens < maximum_rows ||
    dispatch.prefill().row_bucket() < maximum_tokens {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  if demand == QueryRowEnds && dispatch.shape().family() != LanguageModelHead {
    raise InvalidProjectionTileCompilation(InvalidDemand)
  }
  { dispatch, maximum_rows, maximum_tokens, demand }
}

///|
fn projection_tile_value_id(value : Int) -> ProjectionTileValueId {
  ProjectionTileValueId(value)
}
