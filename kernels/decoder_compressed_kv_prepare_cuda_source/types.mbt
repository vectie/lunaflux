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
  priv window_size : Int
  priv compression_ratio : Int
  priv head_dimension : Int
  priv maximum_position_count : Int
  priv maximum_compressed_slots : Int
  priv maximum_key_slots : Int
  priv maximum_batches : Int
  priv function_symbol : String
  priv source : Bytes
}
