///|
/// MiniCPM5's number isolation followed by its word split is equivalent to the
/// digit-triplet byte-BPE profile. Its TemplateProcessing prepends BOS id zero.
/// The chat renderer must omit the textual <s> because this owner inserts it.
pub fn TokenizerSpec::new_minicpm5_bpe(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
) -> TokenizerSpec raise TokenizerError {
  let spec = new_bytelevel_bpe(
    vocabulary,
    merges,
    digest,
    limits,
    profile=TOKENIZER_PROFILE_DIGIT_TRIPLET_BPE,
    ignore_merges=false,
  )
  guard spec.tokens_by_id.get(0) is Some(token) &&
    token.kind is Special &&
    token.piece == b"<s>" else {
    raise InvalidTokenId(entry_index=0)
  }
  { ..spec, post_processor_token_id: 0, }
}
