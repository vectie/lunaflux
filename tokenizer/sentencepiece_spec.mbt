///|
const SENTENCEPIECE_META_SPACE_0 : Byte = b'\xe2'

///|
const SENTENCEPIECE_META_SPACE_1 : Byte = b'\x96'

///|
const SENTENCEPIECE_META_SPACE_2 : Byte = b'\x81'

///|
fn sentencepiece_decoded_piece(
  piece : Bytes,
  entry_index : Int,
) -> Bytes raise TokenizerError {
  let text = @utf8.decode(piece) catch {
    _ => raise InvalidSentencePieceToken(entry_index~)
  }
  ignore(text)
  let decoded : Array[Byte] = Array::new(capacity=piece.length())
  let mut offset = 0
  while offset < piece.length() {
    if offset + 2 < piece.length() &&
      piece[offset] == SENTENCEPIECE_META_SPACE_0 &&
      piece[offset + 1] == SENTENCEPIECE_META_SPACE_1 &&
      piece[offset + 2] == SENTENCEPIECE_META_SPACE_2 {
      decoded.push(b' ')
      offset += 3
    } else {
      decoded.push(piece[offset])
      offset += 1
    }
  }
  Bytes::from_array(decoded)
}

///|
fn sentencepiece_is_one_scalar(piece : Bytes) -> Bool {
  let text = try @utf8.decode(piece) catch {
    _ => return false
  } noraise {
    value => value
  }
  let mut count = 0
  for _ in text {
    count += 1
  }
  count == 1
}

///|
fn sentencepiece_symbol_capacity(
  input_capacity : Int,
) -> Int raise TokenizerError {
  if input_capacity <= 0 || input_capacity > (1_073_741_824 - 4) / 6 {
    raise LunaInvalidWorkerCapacity
  }
  input_capacity * 6 + 4
}

///|
fn TokenizerSpec::worker_symbol_capacity(
  self : TokenizerSpec,
  input_capacity : Int,
) -> Int raise TokenizerError {
  if input_capacity <= 0 || input_capacity > self.limits.max_input_bytes {
    raise LunaInvalidWorkerCapacity
  }
  if self.profile == TOKENIZER_PROFILE_SENTENCEPIECE_BPE {
    sentencepiece_symbol_capacity(input_capacity)
  } else if self.profile == TOKENIZER_PROFILE_QWEN3_BPE ||
    self.profile == TOKENIZER_PROFILE_QWEN35_BPE {
    if input_capacity > (1_073_741_824 - 1) / QWEN3_NFC_MAX_UTF8_EXPANSION {
      raise LunaInvalidWorkerCapacity
    }
    input_capacity * QWEN3_NFC_MAX_UTF8_EXPANSION + 1
  } else {
    input_capacity
  }
}

///|
/// Exact startup-preallocated integer cells for one tokenizer worker using
/// this admitted tokenizer profile and caller input envelope.
pub fn TokenizerSpec::luna_worker_required_int_cells(
  self : TokenizerSpec,
  input_capacity : Int,
) -> UInt64 raise TokenizerError {
  self.worker_symbol_capacity(input_capacity).to_uint64() * 3UL
}

///|
/// Construct the one closed SentencePiece-derived BPE profile supported by
/// LunaFlux. Normal pieces are UTF-8 model symbols, byte-fallback entries
/// contain exactly one decoded byte, and the template prepends one BOS token.
/// The decoder replaces U+2581 with space, fuses pieces, and strips one leading
/// ASCII space from a complete output stream.
pub fn TokenizerSpec::new_sentencepiece_bpe(
  vocabulary : ArrayView[VocabularyEntry],
  merges : ArrayView[MergeRule],
  digest : TokenizerDigest,
  limits : TokenizerLimits,
  unknown_token_id~ : Int,
  bos_token_id~ : Int,
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
  let special_tokens : Array[StoredToken] = []
  let initial_scalar_ids : Array[Int] = []
  let byte_token_ids = FixedArray::make(256, -1)
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
    let decoded_piece = match entry.kind {
      ByteFallback => {
        if entry.piece.length() != 1 {
          raise InvalidSentencePieceToken(entry_index~)
        }
        let value = entry.piece[0].to_int()
        if byte_token_ids[value] >= 0 {
          raise DuplicateTokenPiece(
            first_index=byte_token_ids[value],
            duplicate_index=entry_index,
          )
        }
        byte_token_ids[value] = entry.token_id
        entry.piece
      }
      Normal => {
        match piece_first_index.get(entry.piece) {
          Some(first_index) =>
            raise DuplicateTokenPiece(first_index~, duplicate_index=entry_index)
          None => piece_first_index[entry.piece] = entry_index
        }
        let decoded = sentencepiece_decoded_piece(entry.piece, entry_index)
        if sentencepiece_is_one_scalar(entry.piece) {
          initial_scalar_ids.push(entry.token_id)
        }
        decoded
      }
      Special => {
        match piece_first_index.get(entry.piece) {
          Some(first_index) =>
            raise DuplicateTokenPiece(first_index~, duplicate_index=entry_index)
          None => piece_first_index[entry.piece] = entry_index
        }
        ignore(
          @utf8.decode(entry.piece) catch {
            _ => raise InvalidSentencePieceToken(entry_index~)
          },
        )
        entry.piece
      }
      Added => raise InvalidSentencePieceToken(entry_index~)
    }
    let stored : StoredToken = {
      token_id: entry.token_id,
      piece: entry.piece,
      decoded_piece,
      kind: entry.kind,
    }
    tokens_by_id[entry.token_id] = stored
    if entry.kind is Special {
      special_tokens.push(stored)
    }
  }
  if special_tokens.length() > limits.max_special_tokens {
    raise TooManySpecialTokens(
      actual=special_tokens.length(),
      maximum=limits.max_special_tokens,
    )
  }
  for value in 0..<256 {
    if byte_token_ids[value] < 0 {
      raise MissingByteToken(byte_value=value)
    }
  }
  guard tokens_by_id.get(unknown_token_id) is Some(unknown) &&
    unknown.kind is Special else {
    raise InvalidSentencePieceUnknownToken(token_id=unknown_token_id)
  }
  guard tokens_by_id.get(bos_token_id) is Some(bos) && bos.kind is Special else {
    raise InvalidSentencePieceBosToken(token_id=bos_token_id)
  }
  let merge_left_ids = FixedArray::make(merges.length(), 0)
  let merge_right_ids = FixedArray::make(merges.length(), 0)
  let merge_result_ids = FixedArray::make(merges.length(), 0)
  let merge_left_offsets = FixedArray::make(vocabulary.length() + 1, 0)
  let merge_right_sorted = FixedArray::make(merges.length(), 0)
  let merge_rank_sorted = FixedArray::make(merges.length(), 0)
  let pair_first_rule : Map[Int, Map[Int, Int]] = Map([])
  for rule_index, rule in merges {
    guard tokens_by_id.get(rule.left_token_id) is Some(left) &&
      tokens_by_id.get(rule.right_token_id) is Some(right) &&
      tokens_by_id.get(rule.result_token_id) is Some(result) else {
      raise InvalidMergeToken(rule_index~)
    }
    if !(left.kind is Normal) ||
      !(right.kind is Normal) ||
      !(result.kind is Normal) {
      raise SpecialTokenInMerge(rule_index~)
    }
    if !pieces_equal_joined(left.piece, right.piece, result.piece) {
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
  initial_scalar_ids.sort_by((left, right) => {
    Bytes::compare(tokens_by_id[left].piece, tokens_by_id[right].piece)
  })
  let meta : Bytes = [
    SENTENCEPIECE_META_SPACE_0,
    SENTENCEPIECE_META_SPACE_1,
    SENTENCEPIECE_META_SPACE_2,
  ]
  let mut meta_id = -1
  for token_id in initial_scalar_ids {
    if tokens_by_id[token_id].piece == meta {
      meta_id = token_id
    }
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
    profile: TOKENIZER_PROFILE_SENTENCEPIECE_BPE,
    initial_scalar_ids: FixedArray::from_array(initial_scalar_ids),
    sentencepiece_meta_token_id: meta_id,
    post_processor_token_id: bos_token_id,
    decoder_strip_initial_space: true,
    vocab_size: vocabulary.length(),
    merge_count: merges.length(),
  }
}
