///|
fn parse_decode_ids(source : String, maximum : Int) -> Array[Int] raise {
  guard maximum > 0 && source.length() <= 1048576 else {
    fail("invalid decoded-token envelope")
  }
  if source.is_empty() {
    return []
  }
  let ids : Array[Int] = []
  for piece in source.split(",") {
    guard ids.length() < maximum else {
      fail("decoded tokens exceed model context")
    }
    let id = @string.parse_int(piece)
    guard id >= 0 else { fail("negative decoded token") }
    ids.push(id)
  }
  ids
}

///|
async fn decode_text_tokens(
  root : @approved_fs.ApprovedRoot,
  model_root : String,
  label : String,
  source : String,
  output : String,
) -> Unit {
  let config = inspect_config(root)
  let ids = parse_decode_ids(source, config.max_position_embeddings())
  let tokenizer = @text_input.load_checkpoint_tokenizer(
    root,
    model_root,
    "tokenizer.json",
    label,
    vocabulary_size=config.vocabulary_size(),
    maximum_tokens=ids.length().max(1),
  )
  let bytes = tokenizer.decode_bytes(ids, special_policy=Preserve)
  @fs.write_file(output, bytes, create_mode=CreateNew)
  println(
    "decoded_tokens=\{ids.length()} decoded_bytes=\{bytes.length()} payload_hashing=false",
  )
}
