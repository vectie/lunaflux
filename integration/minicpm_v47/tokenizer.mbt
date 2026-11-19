///|
/// The checkpoint's tokenizer.json uses NFC and the single-digit regex.
/// Its tokenizer_config.json has a different descriptive regex; execution
/// follows tokenizer.json. Marker joins stay with this outer model adapter.
pub fn prepare_tokenizer(
  model : @model.Spec,
  document : BytesView,
  label : @tokenizer.TokenizerDigest,
  limits : @tokenizer.TokenizerLimits,
) -> @tokenizer.TokenizerSpec raise {
  let tokenizer = @json_reader.parse_nfc_single_digit_bytes(
    document,
    digest=label,
    limits~,
    max_bytes=33554432,
    max_depth=32,
  )
  guard tokenizer.vocab_size() == model.decoder().vocabulary_size() else {
    fail("MiniCPM tokenizer vocabulary differs from the model")
  }
  let markers = model.markers()
  for
    (id, content) in [
      (248044, "<|endoftext|>"),
      (248045, "<|im_start|>"),
      (248046, "<|im_end|>"),
      (markers.image, "<|image_pad|>"),
      (markers.video, "<|video_pad|>"),
      (markers.image_start, "<image>"),
      (markers.image_end, "</image>"),
      (markers.slice_start, "<slice>"),
      (markers.slice_end, "</slice>"),
      (markers.newline, "\n"),
    ] {
    guard tokenizer.decode_text([id], special_policy=Preserve) == content else {
      fail("MiniCPM tokenizer marker differs from the model")
    }
  }
  tokenizer
}
