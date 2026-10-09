///|
pub(all) enum ChannelMajorProjectionOperation {
  F32PointwiseConv1d
  F32WeightNormalizedConv1d
  F32WeightNormalizedDilatedConv1d
  F32WeightNormalizedConvTranspose1d
} derive(Debug, Eq)

///|
pub(all) enum ChannelMajorProjectionNumericalContract {
  OrderedF32BiasAccumulationF32OutputV1
  OrderedF32PerOutputWeightNormConv1dV1
  OrderedF32PerOutputWeightNormBiasFreeConv1dV1
  OrderedF32PerOutputWeightNormDilatedConv1dV1
  OrderedF32PerInputWeightNormConvTranspose1dV1
} derive(Debug, Eq)

///|
pub(all) suberror ChannelMajorProjectionCudaSourceError {
  InvalidBatchCount
  InvalidChannelCount
  InvalidTimeCount
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
pub struct ChannelMajorProjectionCudaSource {
  priv operation : ChannelMajorProjectionOperation
  priv numerical_contract : ChannelMajorProjectionNumericalContract
  priv batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv time_count : Int
  priv input_elements : Int64
  priv output_elements : Int64
  priv function_symbol : String
  priv source : Bytes
}

///|
pub struct WeightNormalizedConv1dCudaSource {
  priv operation : ChannelMajorProjectionOperation
  priv numerical_contract : ChannelMajorProjectionNumericalContract
  priv batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv input_time : Int
  priv output_time : Int
  priv kernel_size : Int
  priv padding : Int
  priv stride : Int
  priv input_elements : Int64
  priv weight_elements : Int64
  priv output_elements : Int64
  priv normalize_symbol : String
  priv convolution_symbol : String
  priv source : Bytes
}

///|
pub struct WeightNormalizedDilatedConv1dCudaSource {
  priv operation : ChannelMajorProjectionOperation
  priv numerical_contract : ChannelMajorProjectionNumericalContract
  priv batch_count : Int
  priv channel_count : Int
  priv time_count : Int
  priv kernel_size : Int
  priv dilation : Int
  priv padding : Int
  priv element_count : Int64
  priv weight_elements : Int64
  priv normalize_symbol : String
  priv convolution_symbol : String
  priv source : Bytes
}

///|
pub struct WeightNormalizedConvTranspose1dCudaSource {
  priv operation : ChannelMajorProjectionOperation
  priv numerical_contract : ChannelMajorProjectionNumericalContract
  priv batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv input_time : Int
  priv output_time : Int
  priv kernel_size : Int
  priv padding : Int
  priv stride : Int
  priv input_elements : Int64
  priv weight_elements : Int64
  priv output_elements : Int64
  priv normalize_symbol : String
  priv convolution_symbol : String
  priv source : Bytes
}
