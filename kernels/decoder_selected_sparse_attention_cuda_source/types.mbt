///|
pub(all) enum DecoderSelectedSparseAttentionCudaOperation {
  SharedKeyValueSelectedAttentionBf16
} derive(Debug, Eq)

///|
pub(all) enum DecoderSelectedSparseAttentionNumericalContract {
  SerialOrderedF32SinkDenominatorBf16OutputV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderSelectedSparseAttentionCudaSourceError {
  InvalidQueryHeadCount
  InvalidHeadDimension
  InvalidSelectedWidth
  InvalidMaximumKeySlots
  InvalidMaximumRows
  InvalidMaximumBatches
  InvalidFunctionSymbol
  SizeOverflow
} derive(Debug, Eq)

///|
/// Family-neutral source-only selected sparse attention. It consumes already
/// joined selected indices and a borrowed shared K/V buffer. Index production,
/// K/V preparation or ownership, inverse RoPE, compilation, and qualification
/// are outside this object.
pub struct DecoderSelectedSparseAttentionCudaSource {
  priv operation : DecoderSelectedSparseAttentionCudaOperation
  priv numerical_contract : DecoderSelectedSparseAttentionNumericalContract
  priv query_head_count : Int32
  priv head_dimension : Int32
  priv selected_width : Int32
  priv maximum_key_slots : Int32
  priv maximum_rows : Int32
  priv maximum_batches : Int32
  priv function_symbol : String
  priv source : Bytes
}
