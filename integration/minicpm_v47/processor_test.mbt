///|
async test "actual tokenizer joins slice-aware image fragments and local image IDs" {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(directory) else {
    println(
      "SKIP image marker join: set LUNA_MINICPM_V47_ASSET_DIR to resolved pinned assets",
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
      max_input_bytes: 32768,
      max_output_tokens: 4096,
      max_decoded_bytes: 32768,
      max_vocab_entries: 250000,
      max_merge_rules: 500000,
      max_token_bytes: 2048,
      max_special_tokens: 128,
    },
  )
  for (height, width) in [(448, 448), (896, 448), (896, 896), (1792, 1792)] {
    let plan = @processor.Plan::build(model, height~, width~)
    let fragment = plan.placeholder()
    let encoded = tokenizer.encode_text(
      plan.placeholder_text(),
      special_policy=Allow,
      output_limit=4096,
      truncation=RejectOverflow,
    )
    assert_eq(encoded.token_ids, fragment.tokens()[:].to_owned())
    let prefix = tokenizer.encode_text(
        "<image_id>0</image_id>",
        special_policy=Allow,
        output_limit=32,
        truncation=RejectOverflow,
      ).token_ids
    let prefixed = tokenizer.encode_text(
        plan.placeholder_text(image_id=0),
        special_policy=Allow,
        output_limit=4096,
        truncation=RejectOverflow,
      ).token_ids
    assert_eq(prefixed, prefix + encoded.token_ids)
    let modalities = Array::make(prefix.length(), @positions.Modality::Text)
    for m in fragment.modalities() {
      modalities.push(m)
    }
    let sequence = @positions.Sequence::build(
      model,
      prefixed[:],
      modalities[:],
      image_grids=fragment.grids()[:],
      video_grids=[],
    )
    assert_eq(sequence.prompt_tokens(), prefixed.length())
    assert_true(sequence.rope_delta() < 0)
  }
}
