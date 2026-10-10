///|
// Caller supplies already rendered prompt/template bytes. No implicit chat
// template, BOS, additional normalization or security checksum is inserted here.
async fn prepare_text_frames(
  model_root : String,
  rows : Int,
  history : Int,
  tokenizer_label : String,
  input_path : String,
  output : String,
) -> Unit {
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let config = inspect_config(root)
  guard rows > 0 &&
    rows <= history &&
    history <= config.context_length() &&
    history > 1 else {
    fail("invalid text-frame envelope")
  }
  let digest = @tokenizer.TokenizerDigest::parse(tokenizer_label)
  let tokenizer = @tokenizer_json.parse_bytes(
    read_bounded(root, "tokenizer.json", 67108864L),
    digest~,
    limits={
      max_input_bytes: 1048576,
      max_output_tokens: history - 1,
      max_decoded_bytes: 1048576,
      max_vocab_entries: config.vocabulary_size(),
      max_merge_rules: 2000000,
      max_token_bytes: 4096,
      max_special_tokens: 256,
    },
    max_bytes=67108864,
    max_depth=16,
  )
  guard input_path.has_prefix("/") &&
    input_path.rev_find("/") is Some(separator) else {
    fail("prompt input requires an absolute path")
  }
  let input_root = @approved_fs.ApprovedRoot::open_absolute(
    if separator == 0 {
      "/"
    } else {
      input_path[:separator].to_owned()
    },
  )
  defer input_root.close()
  let bytes = read_bounded(
    input_root,
    input_path[separator + 1:].to_owned(),
    1048576L,
  )
  let encoded = tokenizer.encode_bytes(
    bytes,
    special_policy=Allow,
    output_limit=history - 1,
    truncation=RejectOverflow,
  )
  let (limits, inference) = prompt_limits(
    rows,
    history,
    config.vocabulary_size(),
  )
  let frames = @prompt.encode_prompt(
    encoded.token_ids,
    limits,
    inference,
    maximum_rows=rows,
    maximum_context=history,
    model_generation=1UL,
  )
  let decoded = tokenizer.decode_bytes(
    encoded.token_ids,
    special_policy=Preserve,
  )
  @fs.mkdir(output)
  for index, bytes in frames {
    @fs.write_file(output + "/plan-\{index}.bin", bytes, create_mode=CreateNew)
  }
  @fs.write_file(
    output + "/tokens.txt",
    encoded.token_ids.map(id => id.to_string()).join(",") + "\n",
    create_mode=CreateNew,
  )
  @fs.write_file(output + "/decoded.txt", decoded, create_mode=CreateNew)
  println(
    "prompt_frames=\{frames.length()} input_tokens=\{encoded.token_ids.length()} tokenizer_authentication=false label_kind=caller_supplied",
  )
}
