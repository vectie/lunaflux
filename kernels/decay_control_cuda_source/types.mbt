///|
pub(all) enum DecayControlCudaOperation {
  SafeLowerBoundSigmoidDecayF32
} derive(Debug, Eq)

///|
pub(all) enum DecayControlNumericalContract {
  Bf16LogitF32BiasExpRateOrderedSigmoidF32OutputV1
} derive(Debug, Eq)

///|
pub(all) suberror DecayControlCudaSourceError {
  InvalidHeadCount
  InvalidHeadDimension
  InvalidLowerBound
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
/// Family-neutral pointwise correctness source for a safe negative log-decay
/// transform. It grants no model, compiler, artifact, or execution authority.
pub struct DecayControlCudaSource {
  priv operation : DecayControlCudaOperation
  priv numerical_contract : DecayControlNumericalContract
  priv head_count : Int
  priv head_dimension : Int
  priv lower_bound : Double
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
