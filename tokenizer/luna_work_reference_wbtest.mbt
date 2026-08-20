///|
struct LunaReferenceMerge {
  rank : Int
  result_token_id : Int
}

///|
fn reference_merge_for(
  spec : TokenizerSpec,
  left : Int,
  right : Int,
) -> LunaReferenceMerge? {
  for rank in 0..<spec.merge_count {
    if spec.merge_left_ids[rank] == left && spec.merge_right_ids[rank] == right {
      return Some({ rank, result_token_id: spec.merge_result_ids[rank] })
    }
  }
  None
}

///|
fn reference_encode_ordinary_range(
  spec : TokenizerSpec,
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
    symbols.push(spec.byte_token_ids[input[offset].to_int()])
  }
  while symbols.length() > 1 {
    let mut selected_rank = -1
    let mut selected_left = -1
    let mut selected_right = -1
    for index in 0..<(symbols.length() - 1) {
      match reference_merge_for(spec, symbols[index], symbols[index + 1]) {
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
    let selected = reference_merge_for(spec, selected_left, selected_right).unwrap()
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
fn reference_bytes_match_at(input : Bytes, offset : Int, piece : Bytes) -> Bool {
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
fn reference_special_at(
  spec : TokenizerSpec,
  input : Bytes,
  offset : Int,
) -> StoredToken? {
  let mut best : StoredToken? = None
  for candidate in spec.special_tokens {
    if reference_bytes_match_at(input, offset, candidate.piece) {
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
fn reference_encode_bytes(
  spec : TokenizerSpec,
  input : Bytes,
  special_policy : SpecialTokenPolicy,
  output_limit : Int,
  truncation : TruncationPolicy,
) -> EncodeResult raise TokenizerError {
  if output_limit <= 0 || output_limit > spec.limits.max_output_tokens {
    raise InvalidOutputLimit(
      requested=output_limit,
      maximum=spec.limits.max_output_tokens,
    )
  }
  if input.length() > spec.limits.max_input_bytes {
    raise InputTooLong(
      actual=input.length(),
      maximum=spec.limits.max_input_bytes,
    )
  }
  let token_ids : Array[Int] = []
  if special_policy is AsOrdinary || spec.special_tokens.is_empty() {
    reference_encode_ordinary_range(spec, input, 0, input.length(), token_ids)
  } else {
    let mut ordinary_start = 0
    let mut offset = 0
    while offset < input.length() {
      match reference_special_at(spec, input, offset) {
        Some(special) => {
          reference_encode_ordinary_range(
            spec, input, ordinary_start, offset, token_ids,
          )
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
    reference_encode_ordinary_range(
      spec,
      input,
      ordinary_start,
      input.length(),
      token_ids,
    )
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

///|
test "Luna tokenizer work matches frozen reference over deterministic vectors" {
  let spec = luna_work_corpus_spec()
  let alphabet : FixedArray[Byte] = [
    b'a', b'b', b'c', b'd', b'h', b'e', b'l', b'o', b'<', b'>', b's', b'!',
  ]
  let budgets : FixedArray[Int] = [1, 17, 65536]
  let mut state = 23
  for case_index in 0..<96 {
    state = (state * 73 + 41) % 65521
    let length = 1 + state % 24
    let bytes : Array[Byte] = []
    for _ in 0..<length {
      state = (state * 73 + 41) % 65521
      bytes.push(alphabet[state % alphabet.length()])
    }
    let input = Bytes::from_array(bytes)
    let policy = if case_index % 2 == 0 { Allow } else { AsOrdinary }
    let expected = reference_encode_bytes(
      spec,
      input,
      policy,
      64,
      RejectOverflow,
    )
    let actual = run_luna_work(
      spec,
      input,
      policy,
      64,
      RejectOverflow,
      budgets[case_index % budgets.length()],
    )
    assert_eq(actual.result, expected)
  }
}
