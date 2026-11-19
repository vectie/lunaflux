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
  priv batch_count : Int32
  priv channel_count : Int32
  priv time_count : Int32
  priv ratio : Int32
  priv kernel_size : Int32
  priv upsample_time : Int32
  priv input_element_count : Int64
  priv intermediate_element_count : Int64
  priv output_element_count : Int64
  priv upsample_symbol : String
  priv activation_symbol : String
  priv downsample_symbol : String
  priv source : Bytes
}
