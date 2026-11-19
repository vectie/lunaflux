///|
fn parse_decode_ids(source : String, maximum : Int32) -> Array[Int32] raise {
  guard maximum > 0 && source.length() <= 1048576 else {
    fail("invalid decoded-token envelope")
  }
  if source.is_empty() {
    return []
  }
  let ids : Array[Int32] = []
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
/// Decode a bounded output-ID list through the original checkpoint tokenizer.
/// Preserve special tokens and raw bytes, including incomplete UTF-8 suffixes.
/// No weight, CUDA, checksum or file owner escapes this offline frontend.
pub fn load_checkpoint_output_text(
  root : @approved_fs.ApprovedRoot,
  root_name : String,
  tokenizer_relative : String,
  tokenizer_label : String,
  source : String,
  vocabulary_size~ : Int32,
  maximum_tokens~ : Int32,
) -> CheckpointTextInput raise {
  let ids = parse_decode_ids(source, maximum_tokens)
  let tokenizer = load_checkpoint_tokenizer(
    root,
    root_name,
    tokenizer_relative,
    tokenizer_label,
    vocabulary_size~,
    maximum_tokens=ids.length().max(1),
  )
  {
    ids: ReadOnlyArray::from_array(ids),
    decoded: tokenizer.decode_bytes(ids, special_policy=Preserve),
  }
}
