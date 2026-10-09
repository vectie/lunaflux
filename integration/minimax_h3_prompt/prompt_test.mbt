///|
fn fixture() -> (@tokenizer.TokenizerSpec, @text_encoder.EncoderConfig) raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = Array::makei(256, i => {
    token_id: i,
    piece: Bytes::from_array([i.to_byte()]),
    kind: Normal,
  })
  for
    i, piece in [
      "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
    ] {
    vocabulary.push({
      token_id: 256 + i,
      piece: @utf8.encode(piece),
      kind: Special,
    })
  }
  let spec = @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 4096,
      max_output_tokens: 4096,
      max_decoded_bytes: 8192,
      max_vocab_entries: 512,
      max_merge_rules: 1,
      max_token_bytes: 64,
      max_special_tokens: 16,
    },
  )
  let config = @text_encoder.parse_config(
    @utf8.encode(
      (
        #|{"model_type":"qwen3_vl","image_token_id":258,"video_token_id":259,"vision_start_token_id":256,"vision_end_token_id":257,"text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"head_dim":128,"hidden_act":"silu","hidden_size":5120,"intermediate_size":25600,"max_position_embeddings":262144,"num_attention_heads":64,"num_hidden_layers":64,"num_key_value_heads":8,"rms_norm_eps":1e-6,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"},"rope_theta":5000000,"vocab_size":151936}}
      ),
    ),
    max_bytes=4096,
  )
  (spec, config)
}

///|
test "H3 text is verbatim with no chat or special-token additions" {
  let (tokenizer, config) = fixture()
  let prompt = "  hello\n世界  "
  let result = @minimax_h3_prompt.build(
    tokenizer,
    config,
    prompt~,
    Text,
    max_tokens=1024,
  )
  assert_eq(
    tokenizer.decode_text(result.token_ids[:], special_policy=Preserve),
    prompt,
  )
  assert_true(result.placement is None)
  assert_true(result.token_tags.all(t => t == 1))
}

///|
test "FL2VA keyframes expand exact labels and one slot per merged vision row" {
  let (tokenizer, config) = fixture()
  let grid = @text_encoder.VisualGrid::new(
    kind=Image,
    time=1,
    height=2,
    width=4,
  )
  let result = @minimax_h3_prompt.build(
    tokenizer,
    config,
    prompt=" move ",
    Keyframes([(0, grid), (-1, grid)]),
    max_tokens=1024,
  )
  assert_eq(
    tokenizer.decode_text(result.token_ids[:], special_policy=Preserve),
    "<Picture 1>: <|vision_start|><|image_pad|><|image_pad|><|vision_end|><Picture 2>: <|vision_start|><|image_pad|><|image_pad|><|vision_end|> move ",
  )
  guard result.placement is Some(p) else { fail("missing visual placement") }
  assert_eq(p.token_positions.length(), 4)
  for row in p.token_positions {
    assert_eq(result.token_ids[row], 258)
    assert_eq(result.token_tags[row], 0)
  }
  assert_eq(result.token_tags.filter(t => t == 0).length(), 8)
  try
    @minimax_h3_prompt.build(
      tokenizer,
      config,
      prompt="x",
      Keyframes([(-1, grid), (0, grid)]),
      max_tokens=1024,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("reversed frame semantics")
  }
}

///|
test "Ref2VA preserves mixed reference order and independent ordinals" {
  let (tokenizer, config) = fixture()
  let image = @text_encoder.VisualGrid::new(
    kind=Image,
    time=1,
    height=2,
    width=2,
  )
  let video = @text_encoder.VisualGrid::new(
    kind=Video,
    time=2,
    height=2,
    width=2,
  )
  let result = @minimax_h3_prompt.build(
    tokenizer,
    config,
    prompt="\nprompt",
    References([Audio, Image(image), Video(video, [0.15, 0.25]), Audio]),
    max_tokens=1024,
  )
  assert_eq(
    tokenizer.decode_text(result.token_ids[:], special_policy=Preserve),
    "<Audio 1>: <Picture 1>: <|vision_start|><|image_pad|><|vision_end|><Video 1>: <0.1 seconds><|vision_start|><|video_pad|><|vision_end|><0.2 seconds><|vision_start|><|video_pad|><|vision_end|><Audio 2>: \nprompt",
  )
  guard result.placement is Some(p) else { fail("missing visual placement") }
  assert_eq(p.token_positions.length(), 3)
  assert_eq(result.grids.length(), 2)
  assert_eq(result.token_tags.length(), result.token_ids.length())
  try
    @minimax_h3_prompt.build(
      tokenizer,
      config,
      prompt="x",
      References([Video(video, [0.5])]),
      max_tokens=1024,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("timestamp/grid mismatch")
  }
  try
    @minimax_h3_prompt.build(
      tokenizer,
      config,
      prompt="x",
      References([Image(image)]),
      max_tokens=8,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("token budget ignored")
  }
  try
    @minimax_h3_prompt.build(
      tokenizer,
      config,
      prompt="<|image_pad|>",
      Text,
      max_tokens=1024,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unbound media slot")
  }
}
