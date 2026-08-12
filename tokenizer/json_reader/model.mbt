///|
priv struct ParsedVocabulary {
  entries : Array[@tokenizer.VocabularyEntry]
  normal_id_by_piece : Map[Bytes, Int]
}

///|
fn parse_vocabulary(
  value : Json,
  specials : Map[Int, AddedSpecial],
  limits : @tokenizer.TokenizerLimits,
) -> ParsedVocabulary raise TokenizerJsonError {
  guard value is Object(vocab) else {
    raise InvalidField(component=Model, field=Vocab, index=0)
  }
  if vocab.length() > limits.max_vocab_entries {
    raise VocabularyLimitExceeded(limit=limits.max_vocab_entries)
  }
  let entries : Array[@tokenizer.VocabularyEntry] = []
  let normal_id_by_piece : Map[Bytes, Int] = Map([], capacity=vocab.length())
  let raw_id_by_token : Map[String, Int] = Map([], capacity=vocab.length())
  for raw_piece, id_value in vocab {
    let id = decode_vocab_id(id_value, entries.length())
    let (piece, kind) = match specials.get(id) {
      Some(special) => {
        if special.content != raw_piece {
          raise AddedTokenContentMismatch(index=special.source_index)
        }
        (special.piece, @tokenizer.TokenKind::Special)
      }
      None => (decode_bytelevel_piece(raw_piece, entries.length()), Normal)
    }
    entries.push({ token_id: id, piece, kind })
    raw_id_by_token[raw_piece] = id
    if kind is Normal {
      normal_id_by_piece[piece] = id
    }
  }
  for _, special in specials {
    match raw_id_by_token.get(special.content) {
      Some(id) if id == special.id => ()
      _ => raise AddedTokenNotInVocabulary(index=special.source_index)
    }
  }
  { entries, normal_id_by_piece }
}

///|
fn decode_vocab_id(value : Json, index : Int) -> Int raise TokenizerJsonError {
  guard value is Number(number, repr~) else {
    raise InvalidField(component=Model, field=Vocab, index~)
  }
  let id : Int = match repr {
    Some(encoded) =>
      @strconv.from_str(encoded) catch {
        _ => raise InvalidField(component=Model, field=Vocab, index~)
      }
    None => {
      let integer = number.to_int()
      if number.is_nan() || integer.to_double() != number {
        raise InvalidField(component=Model, field=Vocab, index~)
      }
      integer
    }
  }
  id
}

///|
fn split_merge_row(
  row : String,
  index : Int,
) -> (String, String) raise TokenizerJsonError {
  let mut separator = -1
  for offset in 0..<row.length() {
    if row[offset] == ' ' {
      if separator >= 0 {
        raise MalformedMerge(index~)
      }
      separator = offset
    }
  }
  if separator <= 0 || separator + 1 >= row.length() {
    raise MalformedMerge(index~)
  }
  (row[0:separator].to_owned(), row[separator + 1:row.length()].to_owned())
}

///|
fn decode_merge_pair(
  value : Json,
  index : Int,
) -> (String, String) raise TokenizerJsonError {
  match value {
    String(row) => split_merge_row(row, index)
    Array(pair) => {
      guard pair.length() == 2 &&
        pair[0] is String(left) &&
        pair[1] is String(right) else {
        raise MalformedMerge(index~)
      }
      if left.is_empty() || right.is_empty() {
        raise MalformedMerge(index~)
      }
      (left, right)
    }
    _ => raise MalformedMerge(index~)
  }
}

///|
fn parse_merges(
  value : Json,
  vocabulary : ParsedVocabulary,
  limits : @tokenizer.TokenizerLimits,
) -> Array[@tokenizer.MergeRule] raise TokenizerJsonError {
  guard value is Array(rows) else {
    raise InvalidField(component=Model, field=Merges, index=0)
  }
  if rows.length() > limits.max_merge_rules {
    raise MergeLimitExceeded(limit=limits.max_merge_rules)
  }
  let rules : Array[@tokenizer.MergeRule] = []
  for index, row in rows {
    let (left_raw, right_raw) = decode_merge_pair(row, index)
    let left_piece = decode_bytelevel_piece(left_raw, index)
    let right_piece = decode_bytelevel_piece(right_raw, index)
    let left_id = match vocabulary.normal_id_by_piece.get(left_piece) {
      Some(id) => id
      None => raise UnknownMergePiece(index~, part=Left)
    }
    let right_id = match vocabulary.normal_id_by_piece.get(right_piece) {
      Some(id) => id
      None => raise UnknownMergePiece(index~, part=Right)
    }
    let result_piece = join_bytes(left_piece, right_piece)
    let result_id = match vocabulary.normal_id_by_piece.get(result_piece) {
      Some(id) => id
      None => raise UnknownMergePiece(index~, part=Result)
    }
    rules.push({
      left_token_id: left_id,
      right_token_id: right_id,
      result_token_id: result_id,
    })
  }
  rules
}

///|
fn parse_model(
  value : Json,
  specials : Map[Int, AddedSpecial],
  digest : @tokenizer.TokenizerDigest,
  limits : @tokenizer.TokenizerLimits,
) -> @tokenizer.TokenizerSpec raise TokenizerJsonError {
  guard value is Object(model) else {
    raise InvalidField(component=Root, field=ModelField, index=0)
  }
  validate_fields(model, Model, [
    "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix",
    "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges",
  ])
  if require_string(model, "type", Model, TypeField, 0) != "BPE" {
    raise UnsupportedSemantics(component=Model, feature=ModelKind)
  }
  require_null(model, "dropout", Model, Dropout, MergeDropout)
  require_null(model, "unk_token", Model, UnknownToken, UnknownTokenFallback)
  require_null(
    model,
    "continuing_subword_prefix",
    Model,
    ContinuingSubwordPrefix,
    ContinuingSubwords,
  )
  require_null(
    model,
    "end_of_word_suffix",
    Model,
    EndOfWordSuffix,
    WordSuffixes,
  )
  if require_bool(model, "fuse_unk", Model, FuseUnknown, 0) {
    raise UnsupportedSemantics(component=Model, feature=UnknownTokenFusion)
  }
  if require_bool(model, "byte_fallback", Model, ByteFallback, 0) {
    raise UnsupportedSemantics(component=Model, feature=ByteFallbackBehavior)
  }
  if require_bool(model, "ignore_merges", Model, IgnoreMerges, 0) {
    raise UnsupportedSemantics(component=Model, feature=IgnoreMergeBehavior)
  }
  let vocabulary = parse_vocabulary(
    require_field(model, "vocab", Model, Vocab),
    specials,
    limits,
  )
  let merges = parse_merges(
    require_field(model, "merges", Model, Merges),
    vocabulary,
    limits,
  )
  @tokenizer.TokenizerSpec::new(vocabulary.entries, merges, digest, limits) catch {
    error => raise Validation(error)
  }
}
