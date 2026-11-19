///|
pub fn stop_token_ids() -> Array[Int32] {
  [1, 130073]
}

///|
/// Pinned MiniCPM5 text-only chat without tools or reasoning history. The
/// tokenizer inserts BOS zero, so document_prefix remains empty. The assistant
/// cue matches upstream enable_thinking=false. Model validation is separate.
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
