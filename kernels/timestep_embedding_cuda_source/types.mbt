///|
pub(all) enum TimestepEmbeddingNumericalContract {
  Float32CosSinExpfPeriod10000V1
  OrderedFloat32DenseSiluDenseV1
} derive(Debug, Eq)

///|
pub(all) suberror TimestepEmbeddingCudaSourceError {
  InvalidRowCount
  InvalidFrequencyWidth
  InvalidFunctionSymbol
}

///|
pub struct TimestepEmbeddingCudaSource {
  priv numerical_contract : TimestepEmbeddingNumericalContract
  priv row_count : Int
  priv frequency_width : Int
  priv function_symbol : String
  priv source : Bytes
}
