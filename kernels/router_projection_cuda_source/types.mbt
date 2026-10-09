///|
pub(all) enum RouterProjectionCudaOperation {
  Float32InputBf16WeightFloat32OutputProjection
} derive(Debug, Eq)

///|
pub(all) enum RouterProjectionNumericalContract {
  OrderedFloat32AccumulationNoBiasV1
} derive(Debug, Eq)

///|
pub(all) suberror RouterProjectionCudaSourceError {
  InvalidInputWidth
  InvalidOutputWidth
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
/// Family-neutral source for a correctness-grade F32-input, BF16-weight,
/// F32-output projection. It grants no compilation or execution authority.
pub struct RouterProjectionCudaSource {
  priv operation : RouterProjectionCudaOperation
  priv numerical_contract : RouterProjectionNumericalContract
  priv input_width : Int
  priv output_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
