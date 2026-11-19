///|
test "initial planner prompt trims content and closes the thinking prefix" {
  assert_eq(
    @minicpm_v47.render_text_prompt("  policy\n", "\n move  "),
    "<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nmove<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  )
  assert_eq(@minicpm_v47.stop_token_ids(), [248044, 248046])
}

///|
async test "real flagship tokenizer joins model markers and encodes the planner text path" {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(directory) else {
    println(
      "SKIP real flagship tokenizer: set LUNA_MINICPM_V47_ASSET_DIR to resolved pinned assets",
    )
    return
  }
  let model = @model.Spec::parse(
    @fs.read_file(directory + "/config.json").binary()[:],
  )
  let tokenizer = @minicpm_v47.prepare_tokenizer(
    model,
    @fs.read_file(directory + "/tokenizer.json").binary()[:],
    @tokenizer.TokenizerDigest::parse("a".repeat(64)),
    {
      max_input_bytes: 8192,
      max_output_tokens: 8192,
      max_decoded_bytes: 32768,
      max_vocab_entries: 250000,
      max_merge_rules: 500000,
      max_token_bytes: 2048,
      max_special_tokens: 128,
    },
  )
  assert_eq(tokenizer.vocab_size(), 248144)
  for
    text in [
      "Hello, robot! 123456", "向前走两步。", "é🙂", "<image><|image_pad|></image>",
      "<think>\n\n</think>\n\n{\"skill_id\":\"one\",\"parameters\":{}}",
    ] {
    let encoded = tokenizer.encode_text(
      text,
      special_policy=Allow,
      output_limit=8192,
      truncation=RejectOverflow,
    )
    assert_eq(
      tokenizer.decode_text(encoded.token_ids, special_policy=Preserve),
      text,
    )
  }
  let prompt = @minicpm_v47.render_text_prompt(
    "Select a skill.", "向前走两步。",
  )
  let encoded = tokenizer.encode_text(
    prompt,
    special_policy=Allow,
    output_limit=8192,
    truncation=RejectOverflow,
  )
  assert_eq(encoded.token_ids[0], 248045)
  assert_eq(encoded.token_ids.filter(id => id == 248045).length(), 3)
  assert_eq(encoded.token_ids.filter(id => id == 248046).length(), 2)
  let prepared = @minicpm_v47.prepare_text_prompt(
    model,
    tokenizer,
    "Select a skill.",
    "向前走两步。",
    maximum_tokens=4096,
  )
  assert_eq(prepared.tokens(), ReadOnlyArray::from_array(encoded.token_ids))
  assert_eq(prepared.positions().prompt_tokens(), encoded.token_ids.length())
  assert_eq(prepared.positions().rope_delta(), 0)
  let canvas = FixedArray::make(3, -1)
  let causal = FixedArray::make(1, -1)
  prepared
  .positions()
  .write_decode_into(
    causal_start=encoded.token_ids.length(),
    rows=1,
    canvas,
    causal,
  )
  assert_eq(causal[0], encoded.token_ids.length())
  assert_eq(canvas[:].to_owned(), [causal[0], causal[0], causal[0]])
}
