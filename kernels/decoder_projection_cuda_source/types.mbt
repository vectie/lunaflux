///|
pub(all) enum DecoderProjectionCudaOperation {
  DenseProjectionBf16
  DenseProjectionF32BiasBf16
  DenseProjectionF32BiasF32
  GroupedDenseProjectionBf16V1
} derive(Debug, Eq)

///|
pub(all) enum DecoderProjectionNumericalContract {
  OrderedFloat32AccumulationBf16RoundV1
  OrderedFloat32BiasAccumulationBf16RoundV1
  OrderedFloat32AccumulationF32OutputV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderProjectionCudaSourceError {
  InvalidInputWidth
  InvalidOutputWidth
  InvalidMaximumRows
  InvalidRowCount
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral deterministic dense-projection CUDA source. This is source
/// text only and grants no compiler, artifact, loader, launch, or
/// qualification authority.
pub struct DecoderProjectionCudaSource {
  priv operation : DecoderProjectionCudaOperation
  priv numerical_contract : DecoderProjectionNumericalContract
  priv input_width : Int
  priv output_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral deterministic fixed-row dense-projection CUDA source. The
/// exact row count is embedded in the source identity; this type grants no
/// compiler, artifact, loader, launch, or qualification authority.
pub struct FixedRowsDecoderProjectionCudaSource {
  priv operation : DecoderProjectionCudaOperation
  priv numerical_contract : DecoderProjectionNumericalContract
  priv row_count : Int64
  priv input_width : Int
  priv output_width : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral grouped projection source. Input is `[row,group,input]`,
/// weights are `[group,output,input]`, and output is `[row,group,output]`.
pub struct GroupedDecoderProjectionCudaSource {
  priv operation : DecoderProjectionCudaOperation
  priv numerical_contract : DecoderProjectionNumericalContract
  priv group_count : Int
  priv input_width : Int
  priv output_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
