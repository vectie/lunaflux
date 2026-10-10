///|
/// Construct the no-normalizer ByteLevel BPE profile whose regex isolates
/// contractions, optional-prefix Unicode letter words, one-to-three numeric
/// scalars, punctuation/newlines and whitespace. Input is not NFC-normalized.
/// Added-token matching and exact-pretoken lookup remain model-neutral.
pub fn TokenizerSpec::new_digit_triplet_bpe(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
  ignore_merges? : Bool = false,
) -> TokenizerSpec raise TokenizerError {
  new_bytelevel_bpe(
    vocabulary,
    merges,
    digest,
    limits,
    profile=TOKENIZER_PROFILE_DIGIT_TRIPLET_BPE,
    ignore_merges~,
  )
}
