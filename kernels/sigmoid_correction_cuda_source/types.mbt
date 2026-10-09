///|
pub(all) enum SigmoidCorrectionCudaOperation {
  SigmoidAndCorrectionF32
} derive(Debug, Eq)

///|
pub(all) enum SigmoidCorrectionNumericalContract {
  Float32SigmoidThenCorrectionAddV1
} derive(Debug, Eq)

///|
pub(all) suberror SigmoidCorrectionCudaSourceError {
  InvalidExpertCount
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
/// Family-neutral correctness source for a sigmoid-and-correction boundary.
/// It grants neither compilation nor execution authority.
pub struct SigmoidCorrectionCudaSource {
  priv operation : SigmoidCorrectionCudaOperation
  priv numerical_contract : SigmoidCorrectionNumericalContract
  priv expert_count : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
