///|
pub(all) enum PrefixRmsNormCudaOperation {
  PrefixRmsNormBf16WithOptionalSuffixCopy
} derive(Debug, Eq)

///|
pub(all) enum PrefixRmsNormNumericalContract {
  OrderedFloat32PrefixReductionBf16RoundAndBitExactSuffixCopyV1
} derive(Debug, Eq)

///|
pub(all) suberror PrefixRmsNormCudaSourceError {
  InvalidPrefixWidth
  InvalidSuffixWidth
  InvalidMaximumRows
  InvalidEpsilon
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral source for RMS-normalizing a row prefix while copying an
/// optional suffix to a distinct output. It grants no compilation authority.
pub struct PrefixRmsNormCudaSource {
  priv operation : PrefixRmsNormCudaOperation
  priv numerical_contract : PrefixRmsNormNumericalContract
  priv prefix_width : Int
  priv suffix_width : Int
  priv maximum_rows : Int
  priv epsilon : Double
  priv function_symbol : String
  priv source : Bytes
}
