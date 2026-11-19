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
  priv window_size : Int32
  priv compression_ratio : Int32
  priv maximum_position_count : Int32
  priv maximum_compressed_width : Int32
  priv maximum_joined_width : Int32
  priv maximum_rows : Int32
  priv function_symbol : String
  priv source : Bytes
}
