///|
pub(all) enum DecoderFoundationCudaOperation {
  TokenEmbeddingBf16
  RmsNormBf16
  PairedRmsNormBf16
} derive(Debug, Eq)

///|
pub(all) enum DecoderFoundationNumericalContract {
  BitExactBf16Copy
  OrderedFloat32ReductionBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderFoundationCudaSourceError {
  InvalidVocabularySize
  InvalidRowCount
  InvalidHiddenSize
  InvalidEpsilon
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral deterministic CUDA source. This is source text only: it
/// grants no compiler, artifact, loader, launch, or qualification authority.
pub struct DecoderFoundationCudaSource {
  priv operation : DecoderFoundationCudaOperation
  priv numerical_contract : DecoderFoundationNumericalContract
  priv function_symbol : String
  priv source : Bytes
}
