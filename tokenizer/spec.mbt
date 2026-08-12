///|
priv struct StoredToken {
  token_id : Int
  piece : Bytes
  kind : TokenKind
}

///|
priv struct StoredMerge {
  rank : Int
  result_token_id : Int
}

///|
/// Validated immutable byte-level BPE specification.
///
/// This Phase 1 package deliberately models an already-extracted byte-level
/// BPE vocabulary. It does not claim arbitrary `tokenizer.json` normalizer or
/// pre-tokenizer compatibility.
pub struct TokenizerSpec {
  priv digest : TokenizerDigest
  priv limits : TokenizerLimits
  priv tokens_by_id : Map[Int, StoredToken]
  priv merges_by_left : Map[Int, Map[Int, StoredMerge]]
  priv byte_token_ids : FixedArray[Int]
  priv special_tokens : Array[StoredToken]
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
/// Token IDs and pieces are unique. A merge result must be the concatenation
/// of its normal left and right pieces. Merge array order defines rank.
pub fn TokenizerSpec::new(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
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
    if entry.token_id < 0 {
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
    match piece_first_index.get(entry.piece) {
      Some(first_index) =>
        raise DuplicateTokenPiece(first_index~, duplicate_index=entry_index)
      None => piece_first_index[entry.piece] = entry_index
    }
    let stored : StoredToken = {
      token_id: entry.token_id,
      piece: entry.piece,
      kind: entry.kind,
    }
    tokens_by_id[entry.token_id] = stored
    match entry.kind {
      Normal => normal_ids_by_piece[entry.piece] = entry.token_id
      Special => special_tokens.push(stored)
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
  let merges_by_left : Map[Int, Map[Int, StoredMerge]] = Map([])
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
    if left_token.kind is Special ||
      right_token.kind is Special ||
      result_token.kind is Special {
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
    let by_right = merges_by_left.get_or_init(rule.left_token_id, () => Map([]))
    by_right[rule.right_token_id] = {
      rank: rule_index,
      result_token_id: rule.result_token_id,
    }
  }
  {
    digest,
    limits,
    tokens_by_id,
    merges_by_left,
    byte_token_ids,
    special_tokens,
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
