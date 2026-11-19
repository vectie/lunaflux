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
  priv head_dimension : Int32
  priv pool_size : Int32
  priv maximum_tokens : Int32
  priv pool_capacity : Int32
  priv function_symbol : String
  priv source : Bytes
}
