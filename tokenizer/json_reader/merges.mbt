///|
fn split_merge_row(
  row : String,
  index : Int32,
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
  index : Int32,
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
  profile : TokenizerJsonProfile,
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
    let left_piece = match profile {
      ByteLevelJson
      | Qwen3Json
      | Qwen35Json
      | DigitTripletJson
      | MiniCPM5Json
      | NfcSingleDigitJson
      | OrderedScriptJson => decode_bytelevel_piece(left_raw, index)
      SentencePieceJson(..) => @utf8.encode(left_raw)
    }
    let right_piece = match profile {
      ByteLevelJson
      | Qwen3Json
      | Qwen35Json
      | DigitTripletJson
      | MiniCPM5Json
      | NfcSingleDigitJson
      | OrderedScriptJson => decode_bytelevel_piece(right_raw, index)
      SentencePieceJson(..) => @utf8.encode(right_raw)
    }
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
