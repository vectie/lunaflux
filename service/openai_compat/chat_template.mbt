///|
const LUNA_CHAT_TEMPLATE_MAX_SEGMENT_BYTES : Int = 512

///|
const LUNA_CHAT_TEMPLATE_SEGMENTS : Int = 7

///|
pub struct LunaOpenAIChatTemplate {
  priv storage : FixedArray[Byte]
  priv offsets : FixedArray[Int]
  priv lengths : FixedArray[Int]
  priv max_rendered_prompt_bytes : Int
}

///|
fn luna_chat_template_total(
  segments : FixedArray[FixedArray[Byte]],
) -> Int raise LunaOpenAICompatError {
  if segments.length() != LUNA_CHAT_TEMPLATE_SEGMENTS {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  let mut total = 0
  for segment in segments {
    if segment.length() > LUNA_CHAT_TEMPLATE_MAX_SEGMENT_BYTES ||
      !luna_openai_valid_utf8(segment, 0, segment.length()) ||
      total > 0x7fffffff - segment.length() {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
    }
    total += segment.length()
  }
  total
}

///|
pub fn LunaOpenAIChatTemplate::required_byte_cells(
  system_prefix : FixedArray[Byte],
  system_suffix : FixedArray[Byte],
  user_prefix : FixedArray[Byte],
  user_suffix : FixedArray[Byte],
  assistant_prefix : FixedArray[Byte],
  assistant_suffix : FixedArray[Byte],
  assistant_cue : FixedArray[Byte],
) -> UInt64 raise LunaOpenAICompatError {
  luna_chat_template_total([
    system_prefix, system_suffix, user_prefix, user_suffix, assistant_prefix, assistant_suffix,
    assistant_cue,
  ]).to_uint64()
}

///|
pub fn LunaOpenAIChatTemplate::required_int_cells() -> UInt64 {
  14UL
}

///|
pub fn LunaOpenAIChatTemplate::new(
  inference : @inference.InferenceLimits,
  system_prefix : FixedArray[Byte],
  system_suffix : FixedArray[Byte],
  user_prefix : FixedArray[Byte],
  user_suffix : FixedArray[Byte],
  assistant_prefix : FixedArray[Byte],
  assistant_suffix : FixedArray[Byte],
  assistant_cue : FixedArray[Byte],
  max_rendered_prompt_bytes~ : Int,
) -> Self raise LunaOpenAICompatError {
  let segments : FixedArray[FixedArray[Byte]] = [
    system_prefix, system_suffix, user_prefix, user_suffix, assistant_prefix, assistant_suffix,
    assistant_cue,
  ]
  let total = luna_chat_template_total(segments)
  if max_rendered_prompt_bytes <= 0 ||
    max_rendered_prompt_bytes > inference.max_text_bytes() ||
    total > max_rendered_prompt_bytes {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  let storage : FixedArray[Byte] = FixedArray::make(total, 0)
  let offsets : FixedArray[Int] = FixedArray::make(7, 0)
  let lengths : FixedArray[Int] = FixedArray::make(7, 0)
  let mut cursor = 0
  for index, segment in segments {
    offsets[index] = cursor
    lengths[index] = segment.length()
    for byte_index in 0..<segment.length() {
      storage[cursor] = segment[byte_index]
      cursor += 1
    }
  }
  let template = { storage, offsets, lengths, max_rendered_prompt_bytes }
  template.validate_for(inference)
  template
}

///|
fn LunaOpenAIChatTemplate::validate_for(
  self : Self,
  inference : @inference.InferenceLimits,
) -> Unit raise LunaOpenAICompatError {
  if self.offsets.length() != LUNA_CHAT_TEMPLATE_SEGMENTS ||
    self.lengths.length() != LUNA_CHAT_TEMPLATE_SEGMENTS ||
    self.max_rendered_prompt_bytes <= 0 ||
    self.max_rendered_prompt_bytes > inference.max_text_bytes() {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatTemplate)
  }
  let mut cursor = 0
  for segment in 0..<LUNA_CHAT_TEMPLATE_SEGMENTS {
    let length = self.lengths[segment]
    if length < 0 ||
      length > LUNA_CHAT_TEMPLATE_MAX_SEGMENT_BYTES ||
      self.offsets[segment] != cursor ||
      cursor > self.storage.length() ||
      length > self.storage.length() - cursor ||
      !luna_openai_valid_utf8(self.storage, cursor, length) {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatTemplate)
    }
    cursor += length
  }
  if cursor != self.storage.length() || cursor > self.max_rendered_prompt_bytes {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatTemplate)
  }
}

///|
fn LunaOpenAIChatTemplate::segment_length(self : Self, segment : Int) -> Int {
  self.lengths[segment]
}

///|
fn LunaOpenAIChatTemplate::segment_byte_at(
  self : Self,
  segment : Int,
  index : Int,
) -> Byte {
  self.storage[self.offsets[segment] + index]
}

///|
