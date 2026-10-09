///|
pub(all) enum GlmHybridSparseAttentionCudaOperation {
  SelectedIndexSparseAttentionBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridSparseAttentionNumericalContract {
  SerialOrderedFloat32QkSoftmaxBf16ProbabilityPvBf16OutputV1
} derive(Debug, Eq)

///|
pub(all) suberror GlmHybridSparseAttentionCudaSourceError {
  InvalidQueryHeadCount
  InvalidKeyValueHeadCount
  InvalidQueryKeyDimension
  InvalidValueDimension
  InvalidSelectedWidth
  InvalidMaximumTokens
  InvalidFunctionSymbol
  SizeOverflow
} derive(Debug, Eq)

///|
/// Source-only selected-index sparse attention. Q/K/V projection, index
/// selection, output projection, compilation, and qualification are excluded.
pub struct GlmHybridSparseAttentionCudaSource {
  priv operation : GlmHybridSparseAttentionCudaOperation
  priv numerical_contract : GlmHybridSparseAttentionNumericalContract
  priv query_head_count : Int
  priv key_value_head_count : Int
  priv query_key_dimension : Int
  priv value_dimension : Int
  priv selected_width : Int
  priv maximum_tokens : Int
  priv function_symbol : String
  priv source : Bytes
}
