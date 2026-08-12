///|
fn validate_utf16(text : String) -> Unit raise TokenizerError {
  let units = text.code_units()
  let mut offset = 0
  while offset < units.length() {
    let value = units[offset].to_int()
    if value >= 0xd800 && value <= 0xdbff {
      if offset + 1 >= units.length() {
        raise InvalidUtf16Input(code_unit_offset=offset)
      }
      let next = units[offset + 1].to_int()
      if next < 0xdc00 || next > 0xdfff {
        raise InvalidUtf16Input(code_unit_offset=offset)
      }
      offset += 2
    } else if value >= 0xdc00 && value <= 0xdfff {
      raise InvalidUtf16Input(code_unit_offset=offset)
    } else {
      offset += 1
    }
  }
}

///|
fn TokenizerSpec::merge_for(
  self : TokenizerSpec,
  left : Int,
  right : Int,
) -> StoredMerge? {
  match self.merges_by_left.get(left) {
    Some(by_right) => by_right.get(right)
    None => None
  }
}

///|
fn TokenizerSpec::encode_ordinary_range(
  self : TokenizerSpec,
  input : Bytes,
  start : Int,
  end : Int,
  output : Array[Int],
) -> Unit {
  if start == end {
    return
  }
  let symbols : Array[Int] = Array::new(capacity=end - start)
  for offset in start..<end {
    symbols.push(self.byte_token_ids[input[offset].to_int()])
  }
  while symbols.length() > 1 {
    let mut selected_rank = -1
    let mut selected_left = -1
    let mut selected_right = -1
    for index in 0..<(symbols.length() - 1) {
      match self.merge_for(symbols[index], symbols[index + 1]) {
        Some(merge) =>
          if selected_rank < 0 || merge.rank < selected_rank {
            selected_rank = merge.rank
            selected_left = symbols[index]
            selected_right = symbols[index + 1]
          }
        None => ()
      }
    }
    if selected_rank < 0 {
      break
    }
    let selected = self.merge_for(selected_left, selected_right).unwrap()
    let merged : Array[Int] = Array::new(capacity=symbols.length())
    let mut index = 0
    while index < symbols.length() {
      if index + 1 < symbols.length() &&
        symbols[index] == selected_left &&
        symbols[index + 1] == selected_right {
        merged.push(selected.result_token_id)
        index += 2
      } else {
        merged.push(symbols[index])
        index += 1
      }
    }
    symbols.clear()
    symbols.append(merged)
  }
  output.append(symbols)
}

///|
fn bytes_match_at(input : Bytes, offset : Int, piece : Bytes) -> Bool {
  if piece.length() > input.length() - offset {
    return false
  }
  for index, value in piece {
    if input[offset + index] != value {
      return false
    }
  }
  true
}

///|
fn TokenizerSpec::special_at(
  self : TokenizerSpec,
  input : Bytes,
  offset : Int,
) -> StoredToken? {
  let mut best : StoredToken? = None
  for candidate in self.special_tokens {
    if bytes_match_at(input, offset, candidate.piece) {
      match best {
        None => best = Some(candidate)
        Some(current) =>
          if candidate.piece.length() > current.piece.length() ||
            (
              candidate.piece.length() == current.piece.length() &&
              candidate.token_id < current.token_id
            ) {
            best = Some(candidate)
          }
      }
    }
  }
  best
}

///|
/// Encode valid MoonBit text as UTF-8 bytes and then deterministic byte BPE.
///
/// `output_limit` must be positive and no larger than the spec limit.
pub fn TokenizerSpec::encode_text(
  self : TokenizerSpec,
  text : String,
  special_policy~ : SpecialTokenPolicy,
  output_limit~ : Int,
  truncation~ : TruncationPolicy,
) -> EncodeResult raise TokenizerError {
  validate_utf16(text)
  self.encode_bytes(
    @utf8.encode(text),
    special_policy~,
    output_limit~,
    truncation~,
  )
}

///|
/// Encode arbitrary bytes. Invalid UTF-8 is intentionally accepted because
/// this tokenizer's normalization domain is bytes, not Unicode scalar values.
pub fn TokenizerSpec::encode_bytes(
  self : TokenizerSpec,
  input : Bytes,
  special_policy~ : SpecialTokenPolicy,
  output_limit~ : Int,
  truncation~ : TruncationPolicy,
) -> EncodeResult raise TokenizerError {
  if output_limit <= 0 || output_limit > self.limits.max_output_tokens {
    raise InvalidOutputLimit(
      requested=output_limit,
      maximum=self.limits.max_output_tokens,
    )
  }
  if input.length() > self.limits.max_input_bytes {
    raise InputTooLong(
      actual=input.length(),
      maximum=self.limits.max_input_bytes,
    )
  }
  let token_ids : Array[Int] = []
  if special_policy is AsOrdinary || self.special_tokens.is_empty() {
    self.encode_ordinary_range(input, 0, input.length(), token_ids)
  } else {
    let mut ordinary_start = 0
    let mut offset = 0
    while offset < input.length() {
      match self.special_at(input, offset) {
        Some(special) => {
          self.encode_ordinary_range(input, ordinary_start, offset, token_ids)
          if special_policy is Reject {
            raise SpecialTokenRejected(token_id=special.token_id)
          }
          token_ids.push(special.token_id)
          offset += special.piece.length()
          ordinary_start = offset
        }
        None => offset += 1
      }
    }
    self.encode_ordinary_range(input, ordinary_start, input.length(), token_ids)
  }
  let actual = token_ids.length()
  if actual > output_limit {
    match truncation {
      RejectOverflow => raise OutputTooLong(actual~, maximum=output_limit)
      Truncate => {
        token_ids.truncate(output_limit)
        return { token_ids, input_bytes: input.length(), truncated: true }
      }
    }
  }
  { token_ids, input_bytes: input.length(), truncated: false }
}
