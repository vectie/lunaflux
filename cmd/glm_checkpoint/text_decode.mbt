///|
async fn decode_text_tokens(
  model_root : String,
  label : String,
  source : String,
  output : String,
) -> Unit {
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let config = inspect_config(root)
  let text = @text_input.load_checkpoint_output_text(
    root,
    model_root,
    "tokenizer.json",
    label,
    source,
    vocabulary_size=config.vocabulary_size(),
    maximum_tokens=config.context_length(),
  )
  let bytes = text.decoded_bytes()
  @fs.write_file(output, bytes, create_mode=CreateNew)
  println(
    "decoded_tokens=\{text.token_ids().length()} decoded_bytes=\{bytes.length()} payload_hashing=false",
  )
}
