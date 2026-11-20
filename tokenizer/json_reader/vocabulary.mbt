///|
priv struct ParsedVocabulary {
  entries : Array[@tokenizer.VocabularyEntry]
  normal_id_by_piece : Map[Bytes, Int32]
}

///|
fn parse_vocabulary(
  value : Json,
  specials : Map[Int32, AddedSpecial],
  limits : @tokenizer.TokenizerLimits,
  profile : TokenizerJsonProfile,
) -> ParsedVocabulary raise TokenizerJsonError {
  guard value is Object(vocab) else {
    raise InvalidField(component=Model, field=Vocab, index=0)
  }
  if vocab.length() > limits.max_vocab_entries {
    raise VocabularyLimitExceeded(limit=limits.max_vocab_entries)
  }
  let entries : Array[@tokenizer.VocabularyEntry] = []
  let normal_id_by_piece : Map[Bytes, Int32] = Map([], capacity=vocab.length())
  let raw_id_by_token : Map[String, Int32] = Map([], capacity=vocab.length())
  for raw_piece, id_value in vocab {
    let id = decode_vocab_id(id_value, entries.length())
    let (piece, kind) = match specials.get(id) {
      Some(special) => {
        if special.content != raw_piece {
          raise AddedTokenContentMismatch(index=special.source_index)
        }
        (
          special.piece,
          if special.special {
            @tokenizer.TokenKind::Special
          } else {
            Added
          },
        )
      }
      None =>
        match profile {
          ByteLevelJson
          | Qwen3Json
          | Qwen35Json
          | DigitTripletJson
          | MiniCPM5Json
          | NfcSingleDigitJson
          | OrderedScriptJson =>
            (decode_bytelevel_piece(raw_piece, entries.length()), Normal)
          SentencePieceJson(..) =>
            match decode_sentencepiece_byte_fallback(raw_piece) {
              Some(byte) => ([byte], ByteFallback)
              None => (@utf8.encode(raw_piece), Normal)
            }
        }
    }
    entries.push({ token_id: id, piece, kind, })
    raw_id_by_token[raw_piece] = id
    if kind is Normal {
      normal_id_by_piece[piece] = id
    }
  }
  for _, special in specials {
    match profile {
      DigitTripletJson
      | OrderedScriptJson
      | MiniCPM5Json
      | NfcSingleDigitJson =>
        if raw_id_by_token.get(special.content) is Some(id) && id != special.id {
          raise AddedTokenContentMismatch(index=special.source_index)
        }
      Qwen3Json | Qwen35Json =>
        if raw_id_by_token.contains(special.content) {
          raise AddedTokenContentMismatch(index=special.source_index)
        }
      _ =>
        match raw_id_by_token.get(special.content) {
          Some(id) if id == special.id => ()
          _ => raise AddedTokenNotInVocabulary(index=special.source_index)
        }
    }
  }
  if profile
    is (DigitTripletJson
    | OrderedScriptJson
    | MiniCPM5Json
    | NfcSingleDigitJson) {
    append_external_added_tokens(entries, raw_id_by_token, specials)
  }
  let qwen_first = qwen_profile_first_added(profile)
  let qwen_last = qwen_profile_last_added(profile)
  if qwen_first >= 0 {
    if entries.length() != qwen_first {
      raise InvalidField(component=Model, field=Vocab, index=entries.length())
    }
    for id in qwen_first..<=qwen_last {
      guard specials.get(id) is Some(added) else {
        raise UnsupportedSemantics(
          component=AddedToken,
          feature=Qwen3AddedTokens,
        )
      }
      entries.push({
        token_id: id,
        piece: added.piece,
        kind: if added.special {
          Special
        } else {
          Added
        },
      })
    }
  }
  { entries, normal_id_by_piece, }
}

///|
fn decode_vocab_id(
  value : Json,
  index : Int32,
) -> Int32 raise TokenizerJsonError {
  guard value is Number(number, repr~) else {
    raise InvalidField(component=Model, field=Vocab, index~)
  }
  let id : Int32 = match repr {
    Some(encoded) =>
      @string.parse_int(encoded) catch {
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
fn hex_nibble(code : UInt16) -> Int32? {
  if code >= '0' && code <= '9' {
    Some(code.to_int() - 0x30)
  } else if code >= 'A' && code <= 'F' {
    Some(code.to_int() - 0x41 + 10)
  } else {
    None
  }
}

///|
fn decode_sentencepiece_byte_fallback(piece : String) -> Byte? {
  if piece.length() != 6 ||
    piece[0] != '<' ||
    piece[1] != '0' ||
    piece[2] != 'x' ||
    piece[5] != '>' {
    return None
  }
  guard hex_nibble(piece[3]) is Some(high) && hex_nibble(piece[4]) is Some(low) else {
    return None
  }
  Some(((high << 4) | low).to_byte())
}
