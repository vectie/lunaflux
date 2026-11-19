///|
async test "pinned MiniCPM5 assets build a Llama plan and preserve text with one BOS" {
  guard @env.get_env_var("LUNA_MINICPM5_ASSET_DIR") is Some(directory) else {
    println(
      "SKIP MiniCPM5 assets: set LUNA_MINICPM5_ASSET_DIR containing config.json and tokenizer.json from revision 42433e90dad95123ab87729fd7011e7110629556",
    )
    return
  }
  let label = @spec.ContentDigest::from_sha256("a".repeat(64))
  let metadata = @config_reader.parse_bytes(
    @fs.read_file(directory + "/config.json").binary(),
    label,
    max_bytes=4096,
    max_depth=16,
  )
  let plan = @llama.build_paged(metadata)
  ignore(plan)
  assert_eq(metadata.spec().layer_count(), 42)
  let tokenizer = @json_reader.parse_bytes(
    @fs.read_file(directory + "/tokenizer.json").binary(),
    digest=@tokenizer.TokenizerDigest::parse("b".repeat(64)),
    limits={
      max_input_bytes: 8192,
      max_output_tokens: 8192,
      max_decoded_bytes: 32768,
      max_vocab_entries: 140000,
      max_merge_rules: 300000,
      max_token_bytes: 1024,
      max_special_tokens: 1024,
    },
    max_bytes=16777216,
    max_depth=32,
  )
  assert_eq(tokenizer.vocab_size(), 130560)
  for
    text in [
      "", "1234567890", "Hello, robot!", "向前走两步。", " 123\n你好🙂",
      "<think>\n\n</think>\n\n{\"skill_id\":\"one\",\"parameters\":{}}",
    ] {
    let encoded = tokenizer.encode_text(
      text,
      special_policy=Allow,
      output_limit=8192,
      truncation=RejectOverflow,
    )
    assert_eq(encoded.token_ids[0], 0)
    assert_eq(
      tokenizer.decode_text(encoded.token_ids, special_policy=Skip),
      text,
    )
    assert_eq(encoded.token_ids.filter(id => id == 0).length(), 1)
  }
  let framed = tokenizer.encode_text(
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    special_policy=Allow,
    output_limit=8192,
    truncation=RejectOverflow,
  )
  assert_eq(framed.token_ids.filter(id => id == 130072).length(), 2)
  assert_eq(framed.token_ids.filter(id => id == 130073).length(), 1)
}
