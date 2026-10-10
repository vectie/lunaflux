///|
/// Construct ByteLevel BPE with identity normalization and three ordered
/// isolated splits: numeric triplets, CJK/kana runs, then mark-aware words,
/// punctuation/symbols and whitespace. Split order is semantically significant.
/// Added tokens remain literal; no model/chat template behavior is introduced.
///
/// ```mbt check
/// test "identity normalization preserves literal bytes" {
///   let entries : Array[@tokenizer.VocabularyEntry] = []
///   for byte in 0..<256 {
///     entries.push({
///       token_id: byte,
///       piece: Bytes::make(1, byte.to_byte()),
///       kind: Normal,
///     })
///   }
///   let spec = @tokenizer.TokenizerSpec::new_ordered_script_bpe(
///     entries,
///     [],
///     @tokenizer.TokenizerDigest::parse("a".repeat(64)),
///     {
///       max_input_bytes: 128,
///       max_output_tokens: 128,
///       max_decoded_bytes: 128,
///       max_vocab_entries: 256,
///       max_merge_rules: 1,
///       max_token_bytes: 32,
///       max_special_tokens: 1,
///     },
///   )
///   let tokens = spec.encode_text(
///     "e\u0301",
///     special_policy=AsOrdinary,
///     output_limit=16,
///     truncation=RejectOverflow,
///   )
///   assert_eq(
///     spec.decode_text(tokens.token_ids, special_policy=Preserve),
///     "e\u0301",
///   )
/// }
/// ```
pub fn TokenizerSpec::new_ordered_script_bpe(
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
    profile=TOKENIZER_PROFILE_ORDERED_SCRIPT_BPE,
    ignore_merges~,
  )
}
