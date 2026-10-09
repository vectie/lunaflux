///|
pub(all) enum JointAttentionNumericalContract {
  OrderedF32AccumulationBf16StagedAttentionV1
} derive(Debug, Eq)

///|
pub(all) suberror JointAttentionCudaSourceError {
  InvalidRowCount
  InvalidDocumentBoundaries
  InvalidWidth
  InvalidEpsilon
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
pub struct JointAttentionCudaSource {
  priv numerical_contract : JointAttentionNumericalContract
  priv row_count : Int
  priv hidden_size : Int
  priv head_count : Int
  priv head_dimension : Int
  priv rotary_dimension : Int
  priv applies_rotary : Bool
  priv function_prefix : String
  priv source : Bytes
}
