///|
fn fp8_projection_source(
  symbol : String,
  parameters : Fp8ProjectionParameters,
  maximum_tokens : Int,
) -> Bytes raise Fp8ProjectionAotError {
  match parameters.family {
    QkvProjection => fp8_qkv_source(symbol, parameters, maximum_tokens)
    DenseProjection | LanguageModelHead =>
      fp8_dense_source(symbol, parameters, maximum_tokens)
    GatedMlp => raise Unsupported(CompoundGatedActivationScale)
  }
}
