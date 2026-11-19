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
  priv hidden_size : Int32
  priv intermediate_size : Int32
  priv routed_expert_count : Int32
  priv experts_per_token : Int32
  priv maximum_rows : Int32
  priv function_symbol : String
  priv source : Bytes
}
