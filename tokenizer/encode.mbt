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
/// Encode arbitrary bytes. The raw-byte profile accepts every byte sequence;
/// Unicode-normalizing profiles reject malformed UTF-8 before tokenization.
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
  let capacity = Int::max(input.length(), 1)
  let worker = LunaTokenizerWorker::new(
    self,
    input_capacity=capacity,
    output_capacity=self.limits.max_output_tokens,
    LunaTokenizerStepBudget::new(LUNA_TOKENIZER_MAX_STEP_WORK_UNITS),
  )
  let work = worker.begin_bytes(
    input,
    special_policy~,
    output_limit~,
    truncation~,
  )
  while work.progress() is LunaTokenizerPending {

  }
  let count = work.token_count()
  let storage = FixedArray::make(count, 0)
  let copy_budget = LUNA_TOKENIZER_MAX_STEP_WORK_UNITS
  let mut copied = 0
  while copied < count {
    let length = Int::min(copy_budget, count - copied)
    work.copy_tokens_to(
      storage,
      source_offset=copied,
      destination_offset=copied,
      length~,
    )
    copied += length
  }
  let token_ids : Array[Int] = Array::new(capacity=count)
  for token_id in storage {
    token_ids.push(token_id)
  }
  let truncated = work.was_truncated()
  work.abort()
  { token_ids, input_bytes: input.length(), truncated }
}
