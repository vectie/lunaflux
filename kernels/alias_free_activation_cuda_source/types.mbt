///|
pub(all) enum AliasFreeActivationNumericalContract {
  OrderedF32UpsampleSnakeBetaDownsampleV1
} derive(Debug, Eq)

///|
pub(all) suberror AliasFreeActivationCudaSourceError {
  InvalidGeometry
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
pub struct AliasFreeActivationCudaSource {
  priv numerical_contract : AliasFreeActivationNumericalContract
  priv batch_count : Int
  priv channel_count : Int
  priv time_count : Int
  priv ratio : Int
  priv kernel_size : Int
  priv upsample_time : Int
  priv input_element_count : Int64
  priv intermediate_element_count : Int64
  priv output_element_count : Int64
  priv upsample_symbol : String
  priv activation_symbol : String
  priv downsample_symbol : String
  priv source : Bytes
}
