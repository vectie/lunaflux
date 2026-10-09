///|
pub(all) enum GlmHybridDsaIndexCudaOperation {
  ProjectedIndexScoreStableTopK
  ReuseSelectedIndicesI32
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridDsaIndexNumericalContract {
  SerialOrderedFloat32ReluWeightedScoreStableTopKV1
  BitExactI32Copy
} derive(Debug, Eq)

///|
pub(all) suberror GlmHybridDsaIndexCudaSourceError {
  InvalidHeadCount
  InvalidHeadDimension
  InvalidTopK
  InvalidPoolSize
  InvalidMaximumTokens
  InvalidOutputWidth
  InvalidFunctionSymbol
  SizeOverflow
} derive(Debug, Eq)

///|
/// Source-only DSA projected-score selection or selected-index reuse. It owns
/// no projection, pooling, sparse-attention, compiler, or launch authority.
pub struct GlmHybridDsaIndexCudaSource {
  priv operation : GlmHybridDsaIndexCudaOperation
  priv numerical_contract : GlmHybridDsaIndexNumericalContract
  priv head_count : Int
  priv head_dimension : Int
  priv top_k : Int
  priv pool_size : Int
  priv maximum_tokens : Int
  priv candidate_capacity : Int
  priv output_width : Int
  priv function_symbol : String
  priv source : Bytes
}
