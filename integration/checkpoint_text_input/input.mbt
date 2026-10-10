///|
/// Root-free text result shared by checkpoint diagnostics. This is not a chat
/// template: the installed tokenizer owns normalization and added-token rules.
pub struct CheckpointTextInput {
  priv ids : ReadOnlyArray[Int]
  priv decoded : Bytes
}

///|
pub fn CheckpointTextInput::token_ids(self : Self) -> ArrayView[Int] {
  self.ids[:]
}

///|
pub fn CheckpointTextInput::decoded_bytes(self : Self) -> Bytes {
  self.decoded
}

///|
/// Encoding always rejects overflow. No implicit BOS, EOS or prompt is added.
pub fn encode_text_input(
  spec : @tokenizer.TokenizerSpec,
  input : Bytes,
  maximum_tokens~ : Int,
) -> CheckpointTextInput raise @tokenizer.TokenizerError {
  let encoded = spec.encode_bytes(
    input,
    special_policy=Allow,
    output_limit=maximum_tokens,
    truncation=RejectOverflow,
  )
  let decoded = spec.decode_bytes(encoded.token_ids, special_policy=Preserve)
  { ids: ReadOnlyArray::from_array(encoded.token_ids), decoded, }
}

///|
/// Raw little-endian int32 IDs consumed by checkpoint-backed text encoders.
pub fn CheckpointTextInput::i32_le_bytes(self : Self) -> Bytes {
  Bytes::makei(self.ids.length() * 4, index => {
    ((self.ids[index / 4].reinterpret_as_uint() >> (index % 4 * 8)) & 0xffU).to_byte()
  })
}

///|
fn read_text_input(path : String, maximum : Int) -> Bytes raise {
  guard path.has_prefix("/") && path.rev_find("/") is Some(separator) else {
    fail("text input requires an absolute canonical path")
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(
    if separator == 0 {
      "/"
    } else {
      path[:separator].to_owned()
    },
  )
  defer root.close()
  let file = root.open_file(
    @approved_fs.ApprovedRelativeLocator::new(path[separator + 1:]),
  )
  defer file.close()
  file.read_immutable_snapshot(maximum_bytes=maximum.to_int64())
}

///|
/// Load installed tokenizer and literal text without payload authentication.
/// File/root handles are closed before encoding; only the result escapes.
fn load_text_input(
  root : @approved_fs.ApprovedRoot,
  root_name : String,
  tokenizer_relative : String,
  tokenizer_label : String,
  input_path : String,
  file_limits~ : @tokenizer_file.TokenizerFileLimits,
  limits~ : @tokenizer.TokenizerLimits,
) -> CheckpointTextInput raise {
  let spec = @tokenizer_file.load(
    root,
    root_name,
    tokenizer_relative,
    @tokenizer.TokenizerDigest::parse(tokenizer_label),
    file_limits,
    limits,
  ).spec()
  let input = read_text_input(input_path, limits.max_input_bytes)
  encode_text_input(spec, input, maximum_tokens=limits.max_output_tokens)
}

///|
/// Versioned checkpoint text envelope: 64 MiB JSON, 1 MiB literal text, and
/// output bounded by the caller's model context. Large added-token tables are
/// allowed without special-casing a model family.
pub fn load_checkpoint_text(
  root : @approved_fs.ApprovedRoot,
  root_name : String,
  tokenizer_relative : String,
  tokenizer_label : String,
  input_path : String,
  vocabulary_size~ : Int,
  maximum_tokens~ : Int,
) -> CheckpointTextInput raise {
  load_text_input(
    root,
    root_name,
    tokenizer_relative,
    tokenizer_label,
    input_path,
    file_limits=@tokenizer_file.TokenizerFileLimits::new(
      max_file_bytes=67108864,
      max_json_depth=16,
    ),
    limits={
      max_input_bytes: 1048576,
      max_output_tokens: maximum_tokens,
      max_decoded_bytes: 1048576,
      max_vocab_entries: vocabulary_size,
      max_merge_rules: 2000000,
      max_token_bytes: 4096,
      max_special_tokens: 4096,
    },
  )
}
