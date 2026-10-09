///|
pub(all) enum GatedRmsNormCudaOperation {
  PerVectorSigmoidGatedRmsNormBf16
} derive(Debug, Eq)

///|
pub(all) enum GatedRmsNormNumericalContract {
  OrderedFloat32RmsNormThenSigmoidGateBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) suberror GatedRmsNormCudaSourceError {
  InvalidHeadCount
  InvalidHeadDimension
  InvalidMaximumRows
  InvalidEpsilon
  InvalidFunctionSymbol
  OutputSizeOverflow
} derive(Debug, Eq)

///|
/// Family-neutral serial correctness source for per-vector RMSNorm followed by
/// a pointwise sigmoid gate. It grants no compilation or execution authority.
pub struct GatedRmsNormCudaSource {
  priv operation : GatedRmsNormCudaOperation
  priv numerical_contract : GatedRmsNormNumericalContract
  priv head_count : Int
  priv head_dimension : Int
  priv maximum_rows : Int
  priv epsilon : Double
  priv function_symbol : String
  priv source : Bytes
}
