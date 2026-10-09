///|
pub(all) enum AffineModulationCudaOperation {
  Bf16ShiftScaleToF32
  FinalAdaLnParameterRow
  PackedModalityAdaLnGather
  F32ChannelMajorDestandardization
} derive(Debug, Eq)

///|
pub(all) enum AffineModulationNumericalContract {
  StagedBf16ShiftScaleF32OutputV1
  SiluF32CastBf16OrderedDenseBf16OutputV1
  BitExactBf16PackedModalityGatherV1
  OrderedF32ChannelMajorMultiplyAddV1
} derive(Debug, Eq)

///|
pub(all) suberror AffineModulationCudaSourceError {
  InvalidRowCount
  InvalidWidth
  DimensionOverflow
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
pub struct AffineModulationCudaSource {
  priv operation : AffineModulationCudaOperation
  priv numerical_contract : AffineModulationNumericalContract
  priv row_count : Int64
  priv width : Int
  priv function_symbol : String
  priv source : Bytes
}
