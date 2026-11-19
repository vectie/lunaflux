///|
pub(all) suberror DecoderCompressedKvPrepareCudaSourceError {
  InvalidCompressionRatio
  InvalidHeadDimension
  InvalidMaximumPositionCount
  InvalidMaximumBatches
  InvalidFunctionSymbol
  SizeOverflow
} derive(Debug, Eq)

///|
pub(all) enum DecoderCompressedKvPrepareOperation {
  NonOverlappingSharedKeyValuePrepareBf16V1
} derive(Debug, Eq)

///|
pub(all) enum DecoderCompressedKvPrepareNumericalContract {
  BitExactBf16SegmentCopyV1
} derive(Debug, Eq)

///|
pub struct DecoderCompressedKvPrepareCudaSource {
  priv operation : DecoderCompressedKvPrepareOperation
  priv numerical_contract : DecoderCompressedKvPrepareNumericalContract
  priv window_size : Int32
  priv compression_ratio : Int32
  priv head_dimension : Int32
  priv maximum_position_count : Int32
  priv maximum_compressed_slots : Int32
  priv maximum_key_slots : Int32
  priv maximum_batches : Int32
  priv function_symbol : String
  priv source : Bytes
}
