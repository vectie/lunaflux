///|
// H3 plain-text presentation contains the literal caption, without an LLM chat
// template or implicit BOS. Media-conditioned presentation remains owned by
// integration/minimax_h3_prompt, not this text-only CLI.
async fn prepare_text_ids(
  root_name : String,
  root : @approved_fs.ApprovedRoot,
  args : ArrayView[String],
) -> Unit {
  let config = @text.parse_config(
    read(root, "text_encoder/config.json", 1048576L),
    max_bytes=1048576,
  )
  let maximum_tokens = @string.parse_int(args[3])
  guard maximum_tokens > 0 && maximum_tokens <= config.max_positions else {
    fail("text token limit exceeds encoder context")
  }
  let input = @text_input.load_checkpoint_text(
    root,
    root_name,
    "text_encoder/tokenizer.json",
    args[0],
    args[1],
    vocabulary_size=config.vocabulary_size,
    maximum_tokens~,
  )
  let ids = input.token_ids()
  // Reject image/video placeholder tokens in text-only input before publishing
  // files. The multimodal presentation path owns their feature-row placement.
  ignore(config.plan(ids))
  let output = args[2]
  @fs.mkdir(output)
  @fs.write_file(
    output + "/input-ids.i32",
    input.i32_le_bytes(),
    create_mode=CreateNew,
  )
  @fs.write_file(
    output + "/tokens.txt",
    ids.to_array().map(id => id.to_string()).join(",") + "\n",
    create_mode=CreateNew,
  )
  @fs.write_file(
    output + "/decoded.txt",
    input.decoded_bytes(),
    create_mode=CreateNew,
  )
  println(
    "text_rows=\{ids.length()} tokenizer_authentication=false label_kind=caller_supplied",
  )
}
