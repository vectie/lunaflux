///|
pub(all) enum DecoderInterleavedRopeCudaOperation {
  InterleavedPairRopeWithQueryPrefixCopyBf16
} derive(Debug, Eq)

///|
pub(all) enum DecoderInterleavedRopeNumericalContract {
  Float32DefaultFrequencyTrigRoundedBf16PairArithmeticV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderInterleavedRopeCudaSourceError {
  InvalidHeadCount
  InvalidHeadDimension
  InvalidRopeDimension
  InvalidMaximumRows
  InvalidMaximumPositions
  InvalidTheta
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral default RoPE source for adjacent-pair inputs whose rotated
/// pair components are written in split halves. It grants no AOT authority.
pub struct DecoderInterleavedRopeCudaSource {
  priv operation : DecoderInterleavedRopeCudaOperation
  priv numerical_contract : DecoderInterleavedRopeNumericalContract
  priv query_head_count : Int32
  priv key_head_count : Int32
  priv query_head_dimension : Int32
  priv rope_dimension : Int32
  priv maximum_rows : Int32
  priv maximum_positions : Int32
  priv theta : Double
  priv function_symbol : String
  priv source : Bytes
}
