///|
pub(all) enum GatedMlpNumericalContract {
  OrderedF32AccumulationBf16SwiGluDownBf16V1
  OrderedF32AccumulationClampedBf16SwiGluDownBf16V1
} derive(Debug, Eq)

///|
pub(all) suberror GatedMlpCudaSourceError {
  InvalidRowCount
  InvalidWidth
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
pub struct GatedMlpCudaSource {
  priv numerical_contract : GatedMlpNumericalContract
  priv row_count : Int
  priv hidden_size : Int
  priv intermediate_size : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Counts-bounded family-neutral source for separate gate/up/down matrices.
/// It carries no compiler, artifact, launch, or qualification authority.
pub struct SeparateWeightGatedMlpCudaSource {
  priv numerical_contract : GatedMlpNumericalContract
  priv hidden_size : Int
  priv intermediate_size : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
