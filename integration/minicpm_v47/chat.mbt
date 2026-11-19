///|
pub fn stop_token_ids() -> Array[Int32] {
  [248044, 248046]
}

///|
/// Whitespace removed by Python/Jinja's `trim` in the pinned chat template.
/// The MoonBit default trims only tab/newline/CR/space. Keep this model's format
/// rule at request preparation, including the four C0 separator characters.
///
/// ```mbt check
/// test {
///   assert_eq(@minicpm_v47.trim_message_content("\u{00a0}move\u{3000}"), "move")
/// }
/// ```
pub fn trim_message_content(content : String) -> String {
  content
  .trim(
    chars="\u{0009}\u{000a}\u{000b}\u{000c}\u{000d}\u{001c}\u{001d}\u{001e}\u{001f} \u{0085}\u{00a0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}",
  )
  .to_owned()
}

///|
/// Exact initial system/user text pair, no tools or reasoning, for diagnostics
/// and native-token requests. Mirrors the pinned template's content trimming.
/// The tokenizer adds no BOS; role markers already provide the opening token.
pub fn render_text_prompt(
  system : String,
  user : String,
  system_present~ : Bool = true,
) -> String {
  (if system_present {
    "<|im_start|>system\n" + trim_message_content(system) + "<|im_end|>\n"
  } else {
    ""
  }) +
  "<|im_start|>user\n" +
  trim_message_content(user) +
  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

///|
/// Stateless text-only planner profile. Its caller supplies message strings
/// normalized with `trim_message_content`. Images, tools and reasoning history need
/// the separate multimodal request/processor adapter; this is not that adapter.
pub fn chat_template(
  inference : @inference.InferenceLimits,
  max_rendered_prompt_bytes~ : Int32,
) -> @openai_compat.LunaOpenAIChatTemplate raise @openai_compat.LunaOpenAICompatError {
  fn bytes(text : String) -> FixedArray[Byte] {
    @utf8.encode(text).to_fixedarray()
  }
  @openai_compat.LunaOpenAIChatTemplate::new(
    inference,
    bytes("<|im_start|>system\n"),
    bytes("<|im_end|>\n"),
    bytes("<|im_start|>user\n"),
    bytes("<|im_end|>\n"),
    bytes("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    bytes("<|im_end|>\n"),
    bytes("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    max_rendered_prompt_bytes~,
  )
}
