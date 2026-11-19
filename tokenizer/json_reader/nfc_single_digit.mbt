///|
/// NFC + the exact single-digit word split + byte BPE. Added-token IDs are
/// model data rather than one Qwen vocabulary's fixed suffix. ASCII added
/// tokens avoid a raw-versus-normalized matching-order ambiguity. The model
/// adapter must join vocabulary size and its marker IDs separately.
pub fn parse_nfc_single_digit_bytes(
  document : BytesView,
  digest~ : @tokenizer.TokenizerDigest,
  limits~ : @tokenizer.TokenizerLimits,
  max_bytes~ : Int32,
  max_depth~ : Int32,
) -> @tokenizer.TokenizerSpec raise TokenizerJsonError {
  validate_reader_limits(max_bytes, max_depth)
  if document.length() > max_bytes {
    raise ByteLimitExceeded(limit=max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_text(text, digest, limits, max_depth, nfc_single_digit=true)
}

///|
fn validate_nfc_single_digit_root(
  root : Map[String, Json],
) -> TokenizerJsonProfile raise TokenizerJsonError {
  validate_fields(root, Root, [
    "version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer",
    "post_processor", "decoder", "model",
  ])
  if require_string(root, "version", Root, Version, 0) != "1.0" {
    raise UnsupportedSemantics(component=Root, feature=SchemaVersion)
  }
  require_null(root, "truncation", Root, Truncation, TruncationBehavior)
  require_null(root, "padding", Root, Padding, PaddingBehavior)
  let normalizer = require_component_object(
    require_field(root, "normalizer", Root, NormalizerField),
    Normalizer,
  )
  validate_fields(normalizer, Normalizer, ["type"])
  if require_string(normalizer, "type", Normalizer, TypeField, 0) != "NFC" {
    raise UnsupportedSemantics(
      component=Normalizer,
      feature=NormalizationBehavior,
    )
  }
  let sequence = require_component_object(
    require_field(root, "pre_tokenizer", Root, PreTokenizerField),
    PreTokenizer,
  )
  validate_fields(sequence, PreTokenizer, ["type", "pretokenizers"])
  guard sequence.get("type") is Some(String("Sequence")) &&
    sequence.get("pretokenizers") is Some(Array([split, bytelevel])) else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  guard validate_qwen3_split(split) is Qwen3Json else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  let (prefix, regex) = validate_bytelevel_component(bytelevel, PreTokenizer)
  if prefix || regex {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  // Prefix/regex/trim flags on ByteLevel postprocessing/decoding affect only
  // offsets, not token IDs or decoded byte pieces. No prefix insertion occurs.
  ignore(
    validate_bytelevel_component(
      require_field(root, "post_processor", Root, PostProcessorField),
      PostProcessor,
    ),
  )
  ignore(
    validate_bytelevel_component(
      require_field(root, "decoder", Root, DecoderField),
      Decoder,
    ),
  )
  NfcSingleDigitJson
}
