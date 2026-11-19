///|
pub(all) enum GatedMlpNumericalContract {
  OrderedF32AccumulationBf16SwiGluDownBf16V1
  OrderedF32AccumulationClampedBf16SwiGluDownBf16V1
} derive(Debug, Eq)

///|
pub(all) suberror GatedMlpCudaSourceError {
  UnsupportedDotReductionLaw
  InvalidRowCount
  InvalidWidth
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
pub struct GatedMlpCudaSource {
  priv numerical_contract : GatedMlpNumericalContract
  priv row_count : Int32
  priv hidden_size : Int32
  priv intermediate_size : Int32
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Counts-bounded family-neutral source for separate gate/up/down matrices.
/// It carries no compiler, artifact, launch, or qualification authority.
pub struct SeparateWeightGatedMlpCudaSource {
  priv numerical_contract : GatedMlpNumericalContract
  priv hidden_size : Int32
  priv intermediate_size : Int32
  priv maximum_rows : Int32
  priv function_symbol : String
  priv source : Bytes
}
