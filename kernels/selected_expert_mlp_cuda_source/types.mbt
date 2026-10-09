///|
pub(all) enum SelectedExpertActivation {
  StandardSwiGlu
  BoundedClampedSwiGlu(Double)
} derive(Debug, Eq)

///|
pub(all) enum SelectedExpertMlpNumericalContract {
  ExpertAscendingBf16SwiGluWeightedAccumulateV1
} derive(Debug, Eq)

///|
pub(all) suberror SelectedExpertMlpCudaSourceError {
  InvalidGeometry
  InvalidActivation
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
pub struct SelectedExpertMlpCudaSource {
  priv numerical_contract : SelectedExpertMlpNumericalContract
  priv activation : SelectedExpertActivation
  priv hidden_size : Int
  priv intermediate_size : Int
  priv routed_expert_count : Int
  priv experts_per_token : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
