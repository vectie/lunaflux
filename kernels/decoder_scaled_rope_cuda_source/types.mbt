///|
pub(all) enum DecoderScaledRopeNumericalContract {
  AdjacentPairF32YarnBf16RoundV1
  AdjacentPairInverseF32YarnBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderScaledRopeCudaSourceError {
  InvalidHeadCount
  InvalidHeadDimension
  InvalidRopeDimension
  InvalidMaximumRows
  InvalidMaximumPositions
  InvalidOriginalPositions
  InvalidTheta
  InvalidScaling
  InvalidBeta
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral CUDA source for scaled YaRN rotation of RoPE-only slices.
/// It carries no compiler, artifact, loader, launch, or qualification authority.
pub struct DecoderScaledRopeCudaSource {
  priv numerical_contract : DecoderScaledRopeNumericalContract
  priv query_head_count : Int
  priv key_value_head_count : Int
  priv rope_dimension : Int
  priv maximum_rows : Int
  priv maximum_positions : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral CUDA source for inverse scaled YaRN rotation of one complete
/// attention-head buffer. The non-RoPE prefix is copied bit-exactly.
pub struct DecoderInverseRopeCudaSource {
  priv numerical_contract : DecoderScaledRopeNumericalContract
  priv head_count : Int
  priv head_dimension : Int
  priv rope_dimension : Int
  priv maximum_rows : Int
  priv maximum_positions : Int
  priv function_symbol : String
  priv source : Bytes
}
