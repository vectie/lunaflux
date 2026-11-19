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
  priv query_head_count : Int32
  priv key_value_head_count : Int32
  priv rope_dimension : Int32
  priv maximum_rows : Int32
  priv maximum_positions : Int32
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral CUDA source for inverse scaled YaRN rotation of one complete
/// attention-head buffer. The non-RoPE prefix is copied bit-exactly.
pub struct DecoderInverseRopeCudaSource {
  priv numerical_contract : DecoderScaledRopeNumericalContract
  priv head_count : Int32
  priv head_dimension : Int32
  priv rope_dimension : Int32
  priv maximum_rows : Int32
  priv maximum_positions : Int32
  priv function_symbol : String
  priv source : Bytes
}
