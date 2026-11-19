///|
async fn selected_spec() -> @model.Spec {
  @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
}

///|
async test "explicit image tokens retain canvas coordinates and global feature order" {
  let spec = selected_spec()
  let plan = @processor.Plan::build(spec, height=448, width=448)
  let tokens = [100] + plan.placeholder().tokens()[:].to_owned()
  let prompt = @minicpm_v47_multimodal.prepare_image_tokens(
    spec,
    tokens[:],
    [plan],
    maximum_tokens=4096,
  )
  assert_eq(prompt.prompt().tokens()[:].to_owned(), tokens)
  assert_eq(prompt.feature_rows(), 64)
  assert_eq(prompt.feature_map()[0], -1)
  assert_eq(prompt.feature_map()[1], -1)
  assert_eq(prompt.feature_map()[2], 0)
  assert_eq(prompt.feature_map()[65], 63)
  assert_eq(prompt.feature_map()[66], -1)
  assert_true(prompt.prompt().positions().rope_delta() < 0)
  assert_true(
    try
      @minicpm_v47_multimodal.prepare_image_tokens(
        spec,
        ([spec.markers().image] + tokens)[:],
        [plan],
        maximum_tokens=4096,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @minicpm_v47_multimodal.prepare_image_tokens(
        spec,
        tokens[:],
        [plan],
        maximum_tokens=65,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @minicpm_v47_multimodal.prepare_image_tokens(
        spec,
        tokens[:],
        [],
        maximum_tokens=4096,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
async test "image bridge adds one prefill stage and no decode stages" {
  let spec = selected_spec()
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  let text = @bootstrap.text_program(
    operators,
    @bootstrap.compile_spark_attention(operators),
  )
  let program = @minicpm_v47_multimodal.Program::new(text)
  assert_eq(program.launch_count(Prefill), 925)
  assert_eq(program.launch_count(Decode), 924)
  assert_eq(program.metadata_bytes(), 16388L)
  assert_eq(program.scatter_source().parameter_count(), 6)
}

///|
async test "actual tokenizer matches pinned content-list newlines and local image IDs" {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(directory) else {
    println("SKIP real image prompt tokenizer: set LUNA_MINICPM_V47_ASSET_DIR")
    return
  }
  let spec = selected_spec()
  let tokenizer = @prompt.prepare_tokenizer(
    spec,
    @fs.read_file(directory + "/tokenizer.json").binary()[:],
    @tokenizer.TokenizerDigest::parse("0".repeat(64)),
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
  let first = @processor.Plan::build(spec, height=448, width=448)
  let second = @processor.Plan::build(spec, height=896, width=448)
  let prepared = @minicpm_v47_multimodal.prepare_image_prompt(
    spec,
    tokenizer,
    " policy ",
    [Text("  Look"), Image(first), Text("and compare"), Image(second)],
    maximum_tokens=4096,
  )
  let expected = @prompt.render_text_prompt(
    "policy",
    "  Look\n" +
    first.placeholder_text(image_id=0) +
    "\nand compare\n" +
    second.placeholder_text(image_id=1),
  )
  let ids = tokenizer.encode_text(
      expected,
      special_policy=Allow,
      output_limit=4096,
      truncation=RejectOverflow,
    ).token_ids
  assert_eq(prepared.prompt().tokens()[:].to_owned(), ids)
  assert_eq(prepared.feature_rows(), 258)
  assert_eq(prepared.feature_map().filter(index => index >= 0).length(), 258)
  assert_eq(prepared.image_plans().length(), 2)
  assert_true(
    try
      @minicpm_v47_multimodal.prepare_image_prompt(
        spec,
        tokenizer,
        "policy",
        [Text("<|image_pad|>"), Image(first)],
        maximum_tokens=4096,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
