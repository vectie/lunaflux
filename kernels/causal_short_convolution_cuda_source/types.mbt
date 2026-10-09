///|
pub(all) enum CausalShortConvolutionOperation {
  DepthwiseCausalSiluBf16
} derive(Debug, Eq)

///|
pub(all) enum CausalShortConvolutionNumericalContract {
  OrderedFloat32AccumulationSiluBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) suberror CausalShortConvolutionCudaSourceError {
  InvalidChannelCount
  InvalidKernelSize
  InvalidRowCount
  InvalidSequenceCount
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral, counts-bounded depthwise causal convolution source. It
/// carries no compiler, launch, artifact, cache, or qualification authority.
pub struct CausalShortConvolutionCudaSource {
  priv operation : CausalShortConvolutionOperation
  priv numerical_contract : CausalShortConvolutionNumericalContract
  priv channel_count : Int
  priv kernel_size : Int
  priv maximum_rows : Int
  priv maximum_sequences : Int
  priv function_symbol : String
  priv source : Bytes
}
