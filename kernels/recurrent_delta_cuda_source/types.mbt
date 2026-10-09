///|
pub(all) enum RecurrentDeltaOperation {
  CausalNormalizedRecurrentDeltaBf16F32
} derive(Debug, Eq)

///|
pub(all) enum RecurrentDeltaNumericalContract {
  OrderedFloat32NormalizedDeltaBf16OutputV1
} derive(Debug, Eq)

///|
pub(all) suberror RecurrentDeltaCudaSourceError {
  InvalidHeadCount
  InvalidKeyDimension
  InvalidValueDimension
  InvalidNormalizationEpsilon
  InvalidLogDecayLowerBound
  InvalidRowCount
  InvalidSequenceCount
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral, serial correctness source for a causal normalized recurrent
/// delta transition. It carries no compiler, launch, cache, or qualification
/// authority and is not a throughput implementation.
pub struct RecurrentDeltaCudaSource {
  priv operation : RecurrentDeltaOperation
  priv numerical_contract : RecurrentDeltaNumericalContract
  priv head_count : Int
  priv key_dimension : Int
  priv value_dimension : Int
  priv normalization_epsilon : Double
  priv log_decay_lower_bound : Double
  priv maximum_rows : Int
  priv maximum_sequences : Int
  priv function_symbol : String
  priv source : Bytes
}
