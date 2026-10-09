///|
pub(all) enum DecoderTokenHashCudaOperation {
  TokenHashLookupI32
} derive(Debug, Eq)

///|
pub(all) enum DecoderTokenHashNumericalContract {
  BitExactInt32LookupV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderTokenHashCudaSourceError {
  InvalidVocabularySize
  InvalidExpertsPerToken
  InvalidRoutedExpertCount
  InvalidMaximumRows
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral deterministic I32 token-table lookup source. This grants no
/// compiler, artifact, loader, launch, or qualification authority.
pub struct DecoderTokenHashCudaSource {
  priv operation : DecoderTokenHashCudaOperation
  priv numerical_contract : DecoderTokenHashNumericalContract
  priv vocabulary_size : Int
  priv experts_per_token : Int
  priv routed_expert_count : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}
