///|
pub(all) enum RecurrentAttentionProjectionOperation {
  QkvForgetAndInputGateBf16
} derive(Debug, Eq)

///|
pub(all) enum RecurrentAttentionProjectionNumericalContract {
  OrderedFloat32DenseBf16IntermediateSigmoidV1
} derive(Debug, Eq)

///|
pub(all) suberror RecurrentAttentionProjectionCudaSourceError {
  InvalidHiddenWidth
  InvalidHeadCount
  InvalidHeadDimension
  InvalidControlRank
  InvalidRowCount
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral serial correctness source for recurrent-attention input and
/// control projections. It carries no compiler, launch, or qualification
/// authority.
pub struct RecurrentAttentionProjectionCudaSource {
  priv operation : RecurrentAttentionProjectionOperation
  priv numerical_contract : RecurrentAttentionProjectionNumericalContract
  priv hidden_width : Int
  priv head_count : Int
  priv head_dimension : Int
  priv control_rank : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
