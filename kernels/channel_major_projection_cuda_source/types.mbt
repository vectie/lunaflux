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
  AutocastPointwiseF32AccumulationV1(@precision.AutocastFloat)
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
  priv batch_count : Int32
  priv input_channels : Int32
  priv output_channels : Int32
  priv time_count : Int32
  priv input_elements : Int64
  priv output_elements : Int64
  priv function_symbol : String
  priv source : Bytes
}

///|
pub struct WeightNormalizedConv1dCudaSource {
  priv operation : ChannelMajorProjectionOperation
  priv numerical_contract : ChannelMajorProjectionNumericalContract
  priv batch_count : Int32
  priv input_channels : Int32
  priv output_channels : Int32
  priv input_time : Int32
  priv output_time : Int32
  priv kernel_size : Int32
  priv padding : Int32
  priv stride : Int32
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
  priv batch_count : Int32
  priv channel_count : Int32
  priv time_count : Int32
  priv kernel_size : Int32
  priv dilation : Int32
  priv padding : Int32
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
  priv batch_count : Int32
  priv input_channels : Int32
  priv output_channels : Int32
  priv input_time : Int32
  priv output_time : Int32
  priv kernel_size : Int32
  priv padding : Int32
  priv stride : Int32
  priv input_elements : Int64
  priv weight_elements : Int64
  priv output_elements : Int64
  priv normalize_symbol : String
  priv convolution_symbol : String
  priv source : Bytes
}
