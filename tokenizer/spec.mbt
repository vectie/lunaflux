///|
priv struct StoredToken {
  token_id : Int
  /// Model-side bytes used for special matching and BPE merge validation.
  piece : Bytes
  /// Decoder-side bytes after the admitted tokenizer decoder pipeline.
  decoded_piece : Bytes
  kind : TokenKind
}

///|
const TOKENIZER_PROFILE_RAW_BYTE_BPE : Int = 0

///|
const TOKENIZER_PROFILE_SENTENCEPIECE_BPE : Int = 1

///|
const TOKENIZER_PROFILE_QWEN3_BPE : Int = 2

///|
const TOKENIZER_PROFILE_QWEN35_BPE : Int = 3

///|
/// Validated immutable specification for one admitted tokenizer profile.
///
/// LunaFlux supports its raw ByteLevel-BPE contract, one closed
/// SentencePiece-derived BPE `tokenizer.json` pipeline, the pinned dense Qwen3
/// NFC/regex ByteLevel-BPE pipeline, and the separately pinned Qwen3.5/Qwen3.6
/// NFC/mark-aware ByteLevel-BPE pipeline. This type does not claim
/// compatibility with arbitrary normalizer, template, decoder, or model-family
/// profiles.
pub struct TokenizerSpec {
  priv digest : TokenizerDigest
  priv limits : TokenizerLimits
  priv tokens_by_id : Map[Int, StoredToken]
  priv merge_left_ids : FixedArray[Int]
  priv merge_right_ids : FixedArray[Int]
  priv merge_result_ids : FixedArray[Int]
  priv merge_left_offsets : FixedArray[Int]
  priv merge_right_sorted : FixedArray[Int]
  priv merge_rank_sorted : FixedArray[Int]
  priv byte_token_ids : FixedArray[Int]
  priv special_tokens : Array[StoredToken]
  priv profile : Int
  priv initial_scalar_ids : FixedArray[Int]
  priv sentencepiece_meta_token_id : Int
  priv post_processor_token_id : Int
  priv decoder_strip_initial_space : Bool
  priv vocab_size : Int
  priv merge_count : Int
}

///|
fn validate_positive_limit(
  value : Int,
  field : LimitField,
) -> Unit raise TokenizerError {
  if value <= 0 || value > 1_073_741_824 {
    raise InvalidLimit(field~)
  }
}

///|
fn pieces_equal_joined(left : Bytes, right : Bytes, result : Bytes) -> Bool {
  if left.length() > result.length() ||
    right.length() != result.length() - left.length() {
    return false
  }
  for index, value in left {
    if result[index] != value {
      return false
    }
  }
  for index, value in right {
    if result[left.length() + index] != value {
      return false
    }
  }
  true
}

///|
/// Construct and validate an immutable raw-byte BPE specification.
///
/// The normal vocabulary must contain exactly one entry for every single byte.
/// Token IDs are dense in `0..<vocabulary.length()` and pieces are unique. A
/// merge result must be the concatenation of its normal left and right pieces.
/// Merge array order defines rank.
pub fn TokenizerSpec::new(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
) -> TokenizerSpec raise TokenizerError {
  new_bytelevel_bpe(
    vocabulary,
    merges,
    digest,
    limits,
    profile=TOKENIZER_PROFILE_RAW_BYTE_BPE,
  )
}

///|
/// Construct the exact NFC plus Qwen3-regex ByteLevel-BPE profile.
///
/// JSON admission separately pins the regex, ByteLevel options, and the
/// model's exact added-token vocabulary. This constructor exists so the typed
/// tokenizer package owns execution of that already-admitted profile.
pub fn TokenizerSpec::new_qwen3_bpe(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
) -> TokenizerSpec raise TokenizerError {
  new_bytelevel_bpe(
    vocabulary,
    merges,
    digest,
    limits,
    profile=TOKENIZER_PROFILE_QWEN3_BPE,
  )
}

///|
/// Construct the exact NFC plus mark-aware Qwen3.5/Qwen3.6-regex
/// ByteLevel-BPE profile.
///
/// JSON admission separately pins the 248,044-entry ordinary vocabulary, the
/// 26 exact added tokens, the regex, and all ByteLevel options. Keeping this
/// constructor distinct prevents a dense Qwen3 artifact from being silently
/// reinterpreted under the newer mark-aware split semantics.
pub fn TokenizerSpec::new_qwen35_bpe(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
) -> TokenizerSpec raise TokenizerError {
  new_bytelevel_bpe(
    vocabulary,
    merges,
    digest,
    limits,
    profile=TOKENIZER_PROFILE_QWEN35_BPE,
  )
}

///|
fn new_bytelevel_bpe(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
  profile~ : Int,
) -> TokenizerSpec raise TokenizerError {
  validate_positive_limit(limits.max_input_bytes, InputBytes)
  validate_positive_limit(limits.max_output_tokens, OutputTokens)
  validate_positive_limit(limits.max_decoded_bytes, DecodedBytes)
  validate_positive_limit(limits.max_vocab_entries, VocabularyEntries)
  validate_positive_limit(limits.max_merge_rules, MergeRules)
  validate_positive_limit(limits.max_token_bytes, TokenBytes)
  validate_positive_limit(limits.max_special_tokens, SpecialTokens)
  if vocabulary.length() > limits.max_vocab_entries {
    raise VocabularyTooLarge(
      actual=vocabulary.length(),
      maximum=limits.max_vocab_entries,
    )
  }
  if merges.length() > limits.max_merge_rules {
    raise MergeTableTooLarge(
      actual=merges.length(),
      maximum=limits.max_merge_rules,
    )
  }
  let tokens_by_id : Map[Int, StoredToken] = Map(
    [],
    capacity=vocabulary.length(),
  )
  let piece_first_index : Map[Bytes, Int] = Map(
    [],
    capacity=vocabulary.length(),
  )
  let normal_ids_by_piece : Map[Bytes, Int] = Map(
    [],
    capacity=vocabulary.length(),
  )
  let special_tokens : Array[StoredToken] = []
  for entry_index, entry in vocabulary {
    if entry.token_id < 0 || entry.token_id >= vocabulary.length() {
      raise InvalidTokenId(entry_index~)
    }
    if tokens_by_id.contains(entry.token_id) {
      raise DuplicateTokenId(token_id=entry.token_id)
    }
    if entry.piece.is_empty() {
      raise EmptyTokenPiece(entry_index~)
    }
    if entry.piece.length() > limits.max_token_bytes {
      raise TokenPieceTooLong(
        entry_index~,
        actual=entry.piece.length(),
        maximum=limits.max_token_bytes,
      )
    }
    if entry.kind is ByteFallback {
      raise InvalidSentencePieceToken(entry_index~)
    }
    match piece_first_index.get(entry.piece) {
      Some(first_index) =>
        raise DuplicateTokenPiece(first_index~, duplicate_index=entry_index)
      None => piece_first_index[entry.piece] = entry_index
    }
    let stored : StoredToken = {
      token_id: entry.token_id,
      piece: entry.piece,
      decoded_piece: entry.piece,
      kind: entry.kind,
    }
    tokens_by_id[entry.token_id] = stored
    match entry.kind {
      Normal => normal_ids_by_piece[entry.piece] = entry.token_id
      Added | Special => special_tokens.push(stored)
      ByteFallback => abort("raw byte BPE kind validation invariant")
    }
  }
  if special_tokens.length() > limits.max_special_tokens {
    raise TooManySpecialTokens(
      actual=special_tokens.length(),
      maximum=limits.max_special_tokens,
    )
  }
  let byte_token_ids = FixedArray::make(256, -1)
  for value in 0..<256 {
    let piece : Bytes = [value.to_byte()]
    match normal_ids_by_piece.get(piece) {
      Some(token_id) => byte_token_ids[value] = token_id
      None => raise MissingByteToken(byte_value=value)
    }
  }
  let merge_left_ids = FixedArray::make(merges.length(), 0)
  let merge_right_ids = FixedArray::make(merges.length(), 0)
  let merge_result_ids = FixedArray::make(merges.length(), 0)
  let merge_left_offsets = FixedArray::make(vocabulary.length() + 1, 0)
  let merge_right_sorted = FixedArray::make(merges.length(), 0)
  let merge_rank_sorted = FixedArray::make(merges.length(), 0)
  let pair_first_rule : Map[Int, Map[Int, Int]] = Map([])
  for rule_index, rule in merges {
    let left = tokens_by_id.get(rule.left_token_id)
    let right = tokens_by_id.get(rule.right_token_id)
    let result = tokens_by_id.get(rule.result_token_id)
    guard left is Some(left_token) &&
      right is Some(right_token) &&
      result is Some(result_token) else {
      raise InvalidMergeToken(rule_index~)
    }
    if !(left_token.kind is Normal) ||
      !(right_token.kind is Normal) ||
      !(result_token.kind is Normal) {
      raise SpecialTokenInMerge(rule_index~)
    }
    if !pieces_equal_joined(
        left_token.piece,
        right_token.piece,
        result_token.piece,
      ) {
      raise InvalidMergeResult(rule_index~)
    }
    let prior_by_right = pair_first_rule.get_or_init(rule.left_token_id, () => {
      Map([])
    })
    match prior_by_right.get(rule.right_token_id) {
      Some(first_rule) =>
        raise DuplicateMergePair(first_rule~, duplicate_rule=rule_index)
      None => prior_by_right[rule.right_token_id] = rule_index
    }
    merge_left_ids[rule_index] = rule.left_token_id
    merge_right_ids[rule_index] = rule.right_token_id
    merge_result_ids[rule_index] = rule.result_token_id
    merge_left_offsets[rule.left_token_id + 1] += 1
  }
  for token_id in 0..<vocabulary.length() {
    merge_left_offsets[token_id + 1] += merge_left_offsets[token_id]
  }
  let merge_order : Array[Int] = Array::new(capacity=merges.length())
  for rank in 0..<merges.length() {
    merge_order.push(rank)
  }
  merge_order.sort_by((left_rank, right_rank) => {
    let left_rule = merges[left_rank]
    let right_rule = merges[right_rank]
    if left_rule.left_token_id != right_rule.left_token_id {
      left_rule.left_token_id.compare(right_rule.left_token_id)
    } else {
      left_rule.right_token_id.compare(right_rule.right_token_id)
    }
  })
  for sorted_index, rank in merge_order {
    merge_right_sorted[sorted_index] = merges[rank].right_token_id
    merge_rank_sorted[sorted_index] = rank
  }
  {
    digest,
    limits,
    tokens_by_id,
    merge_left_ids,
    merge_right_ids,
    merge_result_ids,
    merge_left_offsets,
    merge_right_sorted,
    merge_rank_sorted,
    byte_token_ids,
    special_tokens,
    profile,
    initial_scalar_ids: FixedArray::make(0, 0),
    sentencepiece_meta_token_id: -1,
    post_processor_token_id: -1,
    decoder_strip_initial_space: false,
    vocab_size: vocabulary.length(),
    merge_count: merges.length(),
  }
}

///|
/// Artifact identity used to salt model plans and prefix-cache keys.
pub fn TokenizerSpec::digest(self : TokenizerSpec) -> TokenizerDigest {
  self.digest
}

///|
/// Number of validated vocabulary entries.
pub fn TokenizerSpec::vocab_size(self : TokenizerSpec) -> Int {
  self.vocab_size
}

///|
/// Number of validated ranked merge rules.
pub fn TokenizerSpec::merge_count(self : TokenizerSpec) -> Int {
  self.merge_count
}

///|
/// Return a copy of the validated resource limits.
pub fn TokenizerSpec::limits(self : TokenizerSpec) -> TokenizerLimits {
  self.limits
}

///|
/// Whether complete and incremental decode must remove one initial ASCII
/// space after applying the admitted decoder's piece replacement.
pub fn TokenizerSpec::decoder_strips_initial_space(
  self : TokenizerSpec,
) -> Bool {
  self.decoder_strip_initial_space
}
