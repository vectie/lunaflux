///|
pub(all) enum GroupedRoutingCudaOperation {
  GroupedChoiceAndSelectedWeightF32
} derive(Debug, Eq)

///|
pub(all) enum GroupedRoutingNumericalContract {
  SerialF32TopGroupChoiceGatherNormalizeScaleV1
} derive(Debug, Eq)

///|
pub(all) suberror GroupedRoutingCudaSourceError {
  InvalidExpertGeometry
  InvalidGroupGeometry
  InvalidMaximumRows
  InvalidScaling
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
/// Family-neutral serial correctness source for grouped expert routing. It
/// grants no compiler, artifact, launch, or execution authority.
pub struct GroupedRoutingCudaSource {
  priv operation : GroupedRoutingCudaOperation
  priv numerical_contract : GroupedRoutingNumericalContract
  priv routed_expert_count : Int
  priv group_count : Int
  priv group_score_topk : Int
  priv selected_group_count : Int
  priv experts_per_token : Int
  priv normalize : Bool
  priv scaling : Double
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
