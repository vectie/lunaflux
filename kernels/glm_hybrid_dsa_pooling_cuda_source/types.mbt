///|
pub(all) enum GlmHybridDsaPoolingCudaOperation {
  DsaKeyPoolCompressionBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridDsaPoolingNumericalContract {
  StableFloat32SoftmaxBf16ProbabilityOrderedFloat32AccumulationV1
} derive(Debug, Eq)

///|
pub(all) suberror GlmHybridDsaPoolingCudaSourceError {
  InvalidHeadDimension
  InvalidPoolSize
  InvalidMaximumTokens
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Deterministic source for the standalone DSA key-pool compression stage.
/// It does not perform candidate scoring, top-k, or tail selection.
pub struct GlmHybridDsaPoolingCudaSource {
  priv operation : GlmHybridDsaPoolingCudaOperation
  priv numerical_contract : GlmHybridDsaPoolingNumericalContract
  priv head_dimension : Int
  priv pool_size : Int
  priv maximum_tokens : Int
  priv pool_capacity : Int
  priv function_symbol : String
  priv source : Bytes
}
