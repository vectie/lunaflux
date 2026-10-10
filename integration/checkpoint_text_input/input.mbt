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
/// Load installed tokenizer without payload authentication. File handles close
/// before returning the pure tokenizer; no filesystem owner escapes.
fn load_tokenizer(
  root : @approved_fs.ApprovedRoot,
  root_name : String,
  tokenizer_relative : String,
  tokenizer_label : String,
  file_limits~ : @tokenizer_file.TokenizerFileLimits,
  limits~ : @tokenizer.TokenizerLimits,
) -> @tokenizer.TokenizerSpec raise {
  @tokenizer_file.load(
    root,
    root_name,
    tokenizer_relative,
    @tokenizer.TokenizerDigest::parse(tokenizer_label),
    file_limits,
    limits,
  ).spec()
}

///|
/// Shared encode/decode envelope: 64 MiB JSON, 1 MiB literal/decoded text,
/// caller-bounded token count. The label is not a payload checksum.
/// The returned tokenizer remains usable after the filesystem root is closed.
pub fn load_checkpoint_tokenizer(
  root : @approved_fs.ApprovedRoot,
  root_name : String,
  tokenizer_relative : String,
  tokenizer_label : String,
  vocabulary_size~ : Int,
  maximum_tokens~ : Int,
) -> @tokenizer.TokenizerSpec raise {
  load_tokenizer(
    root,
    root_name,
    tokenizer_relative,
    tokenizer_label,
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

///|
/// Encode literal checkpoint text. No implicit template, BOS or EOS is added.
pub fn load_checkpoint_text(
  root : @approved_fs.ApprovedRoot,
  root_name : String,
  tokenizer_relative : String,
  tokenizer_label : String,
  input_path : String,
  vocabulary_size~ : Int,
  maximum_tokens~ : Int,
) -> CheckpointTextInput raise {
  let spec = load_checkpoint_tokenizer(
    root,
    root_name,
    tokenizer_relative,
    tokenizer_label,
    vocabulary_size~,
    maximum_tokens~,
  )
  let input = read_text_input(input_path, 1048576)
  encode_text_input(spec, input, maximum_tokens~)
}
