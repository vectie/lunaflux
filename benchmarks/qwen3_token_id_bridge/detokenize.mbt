///|
/// Request-local incremental decode state for one startup-admitted Qwen3
/// tokenizer. The immutable tokenizer is parsed before listener readiness;
/// token steps only copy already-decoded pieces into fixed request-local
/// storage and never read files or recompute artifact identities.
pub struct Qwen3IncrementalDetokenizer {
  priv spec : @tokenizer.TokenizerSpec
  priv combined : FixedArray[Byte]
  priv output : FixedArray[Byte]
  priv mut pending_length : Int
  priv mut started : Bool
  priv mut finished : Bool
  priv strip_initial_space : Bool
}

///|
pub fn Qwen3IncrementalDetokenizer::new(
  spec : @tokenizer.TokenizerSpec,
) -> Qwen3IncrementalDetokenizer raise Qwen3DetokenizerError {
  let max_piece_bytes = spec.limits().max_token_bytes
  if max_piece_bytes <= 0 || max_piece_bytes > 4096 {
    raise InvalidDetokenizer
  }
  {
    spec,
    combined: FixedArray::make(max_piece_bytes + 3, b'\x00'),
    output: FixedArray::make((max_piece_bytes + 3) * 3, b'\x00'),
    pending_length: 0,
    started: false,
    finished: false,
    strip_initial_space: spec.decoder_strips_initial_space(),
  }
}

///|
pub fn Qwen3IncrementalDetokenizer::max_delta_bytes(
  self : Qwen3IncrementalDetokenizer,
) -> Int {
  self.output.length()
}

///|
fn qwen3_utf8_width(first : Int) -> Int {
  if first <= 0x7f {
    1
  } else if first >= 0xc2 && first <= 0xdf {
    2
  } else if first >= 0xe0 && first <= 0xef {
    3
  } else if first >= 0xf0 && first <= 0xf4 {
    4
  } else {
    0
  }
}

///|
fn qwen3_utf8_second_is_valid(first : Int, second : Int) -> Bool {
  second >= 0x80 &&
  second <= 0xbf &&
  !(first == 0xe0 && second < 0xa0) &&
  !(first == 0xed && second > 0x9f) &&
  !(first == 0xf0 && second < 0x90) &&
  !(first == 0xf4 && second > 0x8f)
}

///|
fn Qwen3IncrementalDetokenizer::append_replacement(
  self : Qwen3IncrementalDetokenizer,
  output_offset : Int,
) -> Int {
  self.output[output_offset] = 0xef
  self.output[output_offset + 1] = 0xbf
  self.output[output_offset + 2] = 0xbd
  output_offset + 3
}

///|
fn Qwen3IncrementalDetokenizer::decode_stable_prefix(
  self : Qwen3IncrementalDetokenizer,
  length : Int,
) -> Bytes {
  let mut input_offset = 0
  let mut output_offset = 0
  while input_offset < length {
    let first = self.combined[input_offset].to_int()
    let width = qwen3_utf8_width(first)
    if width == 1 {
      self.output[output_offset] = self.combined[input_offset]
      input_offset += 1
      output_offset += 1
      continue
    }
    if width == 0 {
      input_offset += 1
      output_offset = self.append_replacement(output_offset)
      continue
    }
    if input_offset + 1 >= length {
      break
    }
    let second = self.combined[input_offset + 1].to_int()
    if !qwen3_utf8_second_is_valid(first, second) {
      input_offset += 1
      output_offset = self.append_replacement(output_offset)
      continue
    }
    let mut continuation_count = 1
    while continuation_count < width - 1 &&
          input_offset + continuation_count + 1 < length &&
          self.combined[input_offset + continuation_count + 1] >= 0x80 &&
          self.combined[input_offset + continuation_count + 1] <= 0xbf {
      continuation_count += 1
    }
    if continuation_count == width - 1 {
      for index in 0..<width {
        self.output[output_offset + index] = self.combined[input_offset + index]
      }
      input_offset += width
      output_offset += width
      continue
    }
    if input_offset + continuation_count + 1 == length {
      break
    }
    input_offset += continuation_count + 1
    output_offset = self.append_replacement(output_offset)
  }
  self.pending_length = length - input_offset
  for index in 0..<self.pending_length {
    self.combined[index] = self.combined[input_offset + index]
  }
  Bytes::makei(output_offset, index => self.output[index])
}

///|
/// Advance by one exact output token. Returned bytes are always valid UTF-8;
/// an incomplete scalar is withheld until a later token completes it. Invalid
/// maximal byte subsequences use the standard U+FFFD replacement scalar.
pub fn Qwen3IncrementalDetokenizer::advance(
  self : Qwen3IncrementalDetokenizer,
  token_id : Int,
) -> Bytes raise Qwen3DetokenizerError {
  if self.finished {
    raise DetokenizerFinished
  }
  let copied = self.spec.copy_decoded_piece(
    token_id,
    self.combined,
    destination_offset=self.pending_length,
    special_policy=Skip,
  ) catch {
    _ => raise TokenDecodeFailed
  }
  let mut length = self.pending_length + copied
  if !self.started && copied > 0 {
    self.started = true
    if self.strip_initial_space && self.combined[0] == b' ' {
      for index in 1..<length {
        self.combined[index - 1] = self.combined[index]
      }
      length -= 1
    }
  }
  self.decode_stable_prefix(length)
}

///|
/// Finish one stream. A trailing incomplete UTF-8 scalar becomes exactly one
/// replacement scalar, matching complete lossy decode of the accumulated byte
/// stream without inventing a token event.
pub fn Qwen3IncrementalDetokenizer::finish(
  self : Qwen3IncrementalDetokenizer,
) -> Bytes raise Qwen3DetokenizerError {
  if self.finished {
    raise DetokenizerFinished
  }
  self.finished = true
  if self.pending_length == 0 {
    return b""
  }
  self.pending_length = 0
  let length = self.append_replacement(0)
  Bytes::makei(length, index => self.output[index])
}
