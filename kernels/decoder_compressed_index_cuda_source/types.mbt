///|
pub(all) enum DecoderCompressedIndexCudaOperation {
  CausalCompleteSlotsI32
} derive(Debug, Eq)

///|
pub(all) enum DecoderCompressedIndexNumericalContract {
  BitExactCausalCompleteSlotsI32V1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderCompressedIndexCudaSourceError {
  InvalidCompressionRatio
  InvalidMaximumPositionCount
  InvalidMaximumRows
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral source for deterministic causal compressed-cache indices.
/// It grants no compiler, artifact, loader, launch, or qualification authority.
pub struct DecoderCompressedIndexCudaSource {
  priv operation : DecoderCompressedIndexCudaOperation
  priv numerical_contract : DecoderCompressedIndexNumericalContract
  priv compression_ratio : Int32
  priv maximum_position_count : Int32
  priv maximum_slots_per_row : Int32
  priv maximum_rows : Int32
  priv function_symbol : String
  priv source : Bytes
}
