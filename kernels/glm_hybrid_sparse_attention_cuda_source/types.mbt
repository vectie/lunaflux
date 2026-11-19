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
  priv query_head_count : Int32
  priv key_value_head_count : Int32
  priv query_key_dimension : Int32
  priv value_dimension : Int32
  priv selected_width : Int32
  priv maximum_tokens : Int32
  priv maximum_queries : Int32
  priv function_symbol : String
  priv source : Bytes
}
