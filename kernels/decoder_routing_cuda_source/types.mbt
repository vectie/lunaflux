///|
pub(all) enum DecoderRoutingCudaOperation {
  StableBiasedTopKI32
  SelectedExpertWeightFinalizeF32
} derive(Debug, Eq)

///|
pub(all) enum DecoderRoutingNumericalContract {
  StableLowerExpertIdTopKV1
  OrderedF32NormalizeScaleV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderRoutingCudaSourceError {
  InvalidRoutedExpertCount
  InvalidExpertsPerToken
  InvalidMaximumRows
  InvalidScaling
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral deterministic routing CUDA source. This grants no compiler,
/// artifact, loader, runtime launch, or qualification authority.
pub struct DecoderRoutingCudaSource {
  priv operation : DecoderRoutingCudaOperation
  priv numerical_contract : DecoderRoutingNumericalContract
  priv routed_expert_count : Int32
  priv experts_per_token : Int32
  priv maximum_rows : Int32
  priv function_symbol : String
  priv source : Bytes
}
