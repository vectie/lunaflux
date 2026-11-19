///|
pub(all) enum JointAttentionNumericalContract {
  OrderedF32AccumulationBf16StagedAttentionV1
  GlobalBf16ProbabilityF32DenominatorStagedAttentionV1
  OnlineBf16ProbabilityF32DenominatorStagedAttentionV1(block_tokens~ : Int32)
} derive(Debug, Eq)

///|
pub(all) suberror JointAttentionCudaSourceError {
  InvalidRowCount
  InvalidDocumentBoundaries
  InvalidWidth
  InvalidEpsilon
  DimensionOverflow
  InvalidFunctionSymbol
  InvalidProbabilityLaw
} derive(Debug, Eq)

///|
pub struct JointAttentionCudaSource {
  priv numerical_contract : JointAttentionNumericalContract
  priv row_count : Int32
  priv hidden_size : Int32
  priv head_count : Int32
  priv head_dimension : Int32
  priv rotary_dimension : Int32
  priv applies_rotary : Bool
  priv function_prefix : String
  priv source : Bytes
}
