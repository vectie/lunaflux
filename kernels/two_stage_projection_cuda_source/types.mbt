///|
pub(all) enum TwoStageProjectionOperation {
  BiasFreeBf16LowRankProjection
} derive(Debug, Eq)

///|
pub(all) enum TwoStageProjectionNumericalContract {
  OrderedFloat32AccumulationBf16IntermediateAndOutputV1
} derive(Debug, Eq)

///|
pub(all) suberror TwoStageProjectionCudaSourceError {
  InvalidInputWidth
  InvalidIntermediateWidth
  InvalidOutputWidth
  InvalidRowCount
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
pub struct TwoStageProjectionCudaSource {
  priv operation : TwoStageProjectionOperation
  priv numerical_contract : TwoStageProjectionNumericalContract
  priv input_width : Int32
  priv intermediate_width : Int32
  priv output_width : Int32
  priv maximum_rows : Int32
  priv function_symbol : String
  priv source : Bytes
}
