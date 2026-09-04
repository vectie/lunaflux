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
  {
    dispatch,
    maximum_rows,
    maximum_tokens,
    demand,
    epilogue: ProjectionStoreRows,
  }
}

///|
pub fn ProjectionAttentionIngressEpilogue::new(
  query_heads~ : Int,
  kv_heads~ : Int,
  head_dimension~ : Int,
  norm_epsilon~ : Double,
  rope_theta~ : Double,
) -> ProjectionAttentionIngressEpilogue raise ProjectionTileCompilerError {
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    norm_epsilon.is_nan() ||
    norm_epsilon.is_inf() ||
    norm_epsilon <= 0.0 ||
    rope_theta.is_nan() ||
    rope_theta.is_inf() ||
    rope_theta <= 0.0 {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  { query_heads, kv_heads, head_dimension, norm_epsilon, rope_theta }
}

///|
pub fn ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
  dispatch~ : @luna_projection_strategy.ProjectionDispatchPlan,
  maximum_rows~ : Int,
  maximum_tokens~ : Int,
  epilogue~ : ProjectionAttentionIngressEpilogue,
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  let request = ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows~,
    maximum_tokens~,
    demand=AllTokenRows,
  )
  let (_, output_width, _) = dispatch.shape().dimensions()
  if dispatch.shape().family() != Qkv ||
    output_width !=
    (epilogue.query_heads + 2 * epilogue.kv_heads) * epilogue.head_dimension {
    raise InvalidProjectionTileCompilation(InvalidDemand)
  }
  { ..request, epilogue: ProjectionAttentionIngress(epilogue) }
}

///|
fn projection_tile_value_id(value : Int) -> ProjectionTileValueId {
  ProjectionTileValueId(value)
}
