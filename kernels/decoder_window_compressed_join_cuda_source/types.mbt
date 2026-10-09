///|
pub(all) enum DecoderWindowCompressedJoinCudaOperation {
  DeterministicWindowCompressedJoinI32V1
} derive(Debug, Eq)

///|
pub(all) enum DecoderWindowCompressedJoinNumericalContract {
  BitExactOfficialWindowThenCompressedI32V1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderWindowCompressedJoinCudaSourceError {
  InvalidWindowSize
  InvalidCompressionRatio
  InvalidMaximumPositionCount
  InvalidMaximumRows
  InvalidFunctionSymbol
  SizeOverflow
} derive(Debug, Eq)

///|
pub struct DecoderWindowCompressedJoinCudaSource {
  priv operation : DecoderWindowCompressedJoinCudaOperation
  priv numerical_contract : DecoderWindowCompressedJoinNumericalContract
  priv window_size : Int
  priv compression_ratio : Int
  priv maximum_position_count : Int
  priv maximum_compressed_width : Int
  priv maximum_joined_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
