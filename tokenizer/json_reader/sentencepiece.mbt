///|
fn sentencepiece_unsupported(
  component : JsonComponent,
  feature : UnsupportedFeature,
) -> Unit raise TokenizerJsonError {
  raise UnsupportedSemantics(component~, feature~)
}

///|
fn require_component_object(
  value : Json,
  component : JsonComponent,
) -> Map[String, Json] raise TokenizerJsonError {
  guard value is Object(object) else {
    raise InvalidField(component~, field=TypeField, index=0)
  }
  object
}

///|
fn validate_sentencepiece_replace(
  value : Json,
  component : JsonComponent,
  pattern : String,
  content : String,
) -> Unit raise TokenizerJsonError {
  let object = require_component_object(value, component)
  validate_fields(object, component, ["type", "pattern", "content"])
  if require_string(object, "type", component, TypeField, 0) != "Replace" ||
    require_string(object, "content", component, Content, 0) != content {
    sentencepiece_unsupported(component, NormalizationBehavior)
  }
  let pattern_object = require_component_object(
    require_field(object, "pattern", component, Pattern),
    component,
  )
  validate_fields(pattern_object, component, ["String"])
  if require_string(pattern_object, "String", component, Pattern, 0) != pattern {
    sentencepiece_unsupported(component, NormalizationBehavior)
  }
}

///|
fn validate_sentencepiece_normalizer(
  value : Json,
) -> Unit raise TokenizerJsonError {
  let object = require_component_object(value, Normalizer)
  validate_fields(object, Normalizer, ["type", "normalizers"])
  if require_string(object, "type", Normalizer, TypeField, 0) != "Sequence" {
    sentencepiece_unsupported(Normalizer, NormalizationBehavior)
  }
  guard require_field(object, "normalizers", Normalizer, Components)
    is Array([prepend, replace]) else {
    sentencepiece_unsupported(Normalizer, NormalizationBehavior)
  }
  let prepend_object = require_component_object(prepend, Normalizer)
  validate_fields(prepend_object, Normalizer, ["type", "prepend"])
  if require_string(prepend_object, "type", Normalizer, TypeField, 0) !=
    "Prepend" ||
    require_string(prepend_object, "prepend", Normalizer, Content, 0) != "▁" {
    sentencepiece_unsupported(Normalizer, NormalizationBehavior)
  }
  validate_sentencepiece_replace(replace, Normalizer, " ", "▁")
}

///|
fn validate_template_item(
  value : Json,
  field_name : String,
  expected_id : String,
  expected_type_id : Int,
) -> Unit raise TokenizerJsonError {
  let wrapper = require_component_object(value, PostProcessor)
  validate_fields(wrapper, PostProcessor, [field_name])
  let item = require_component_object(
    require_field(wrapper, field_name, PostProcessor, Template),
    PostProcessor,
  )
  validate_fields(item, PostProcessor, ["id", "type_id"])
  if require_string(item, "id", PostProcessor, Id, 0) != expected_id ||
    require_exact_int(item, "type_id", PostProcessor, TypeId, 0) !=
    expected_type_id {
    sentencepiece_unsupported(PostProcessor, PostProcessingBehavior)
  }
}

///|
fn validate_sentencepiece_post_processor(
  value : Json,
) -> Int raise TokenizerJsonError {
  let object = require_component_object(value, PostProcessor)
  validate_fields(object, PostProcessor, [
    "type", "single", "pair", "special_tokens",
  ])
  if require_string(object, "type", PostProcessor, TypeField, 0) !=
    "TemplateProcessing" {
    sentencepiece_unsupported(PostProcessor, PostProcessingBehavior)
  }
  guard require_field(object, "single", PostProcessor, Template)
    is Array([single_bos, single_sequence]) else {
    raise UnsupportedSemantics(
      component=PostProcessor,
      feature=PostProcessingBehavior,
    )
  }
  validate_template_item(single_bos, "SpecialToken", "<s>", 0)
  validate_template_item(single_sequence, "Sequence", "A", 0)
  guard require_field(object, "pair", PostProcessor, Template)
    is Array([pair_bos_a, pair_sequence_a, pair_bos_b, pair_sequence_b]) else {
    raise UnsupportedSemantics(
      component=PostProcessor,
      feature=PostProcessingBehavior,
    )
  }
  validate_template_item(pair_bos_a, "SpecialToken", "<s>", 0)
  validate_template_item(pair_sequence_a, "Sequence", "A", 0)
  validate_template_item(pair_bos_b, "SpecialToken", "<s>", 1)
  validate_template_item(pair_sequence_b, "Sequence", "B", 1)
  let special_tokens = require_component_object(
    require_field(object, "special_tokens", PostProcessor, SpecialTokens),
    PostProcessor,
  )
  validate_fields(special_tokens, PostProcessor, ["<s>"])
  let bos = require_component_object(
    require_field(special_tokens, "<s>", PostProcessor, SpecialTokens),
    PostProcessor,
  )
  validate_fields(bos, PostProcessor, ["id", "ids", "tokens"])
  if require_string(bos, "id", PostProcessor, Id, 0) != "<s>" {
    sentencepiece_unsupported(PostProcessor, PostProcessingBehavior)
  }
  guard require_field(bos, "ids", PostProcessor, Id) is Array([id_value]) &&
    decode_vocab_id(id_value, 0) == 1 &&
    require_field(bos, "tokens", PostProcessor, Content)
    is Array([String("<s>")]) else {
    raise UnsupportedSemantics(
      component=PostProcessor,
      feature=PostProcessingBehavior,
    )
  }
  1
}

///|
fn validate_type_only_component(
  value : Json,
  expected : String,
) -> Unit raise TokenizerJsonError {
  let object = require_component_object(value, Decoder)
  validate_fields(object, Decoder, ["type"])
  if require_string(object, "type", Decoder, TypeField, 0) != expected {
    sentencepiece_unsupported(Decoder, DecoderKind)
  }
}

///|
fn validate_sentencepiece_decoder(
  value : Json,
) -> Unit raise TokenizerJsonError {
  let object = require_component_object(value, Decoder)
  validate_fields(object, Decoder, ["type", "decoders"])
  if require_string(object, "type", Decoder, TypeField, 0) != "Sequence" {
    sentencepiece_unsupported(Decoder, DecoderKind)
  }
  guard require_field(object, "decoders", Decoder, Components)
    is Array([replace, byte_fallback, fuse, strip]) else {
    sentencepiece_unsupported(Decoder, DecoderKind)
  }
  validate_sentencepiece_replace(replace, Decoder, "▁", " ")
  validate_type_only_component(byte_fallback, "ByteFallback")
  validate_type_only_component(fuse, "Fuse")
  let strip_object = require_component_object(strip, Decoder)
  validate_fields(strip_object, Decoder, ["type", "content", "start", "stop"])
  if require_string(strip_object, "type", Decoder, TypeField, 0) != "Strip" ||
    require_string(strip_object, "content", Decoder, Content, 0) != " " ||
    require_exact_int(strip_object, "start", Decoder, Start, 0) != 1 ||
    require_exact_int(strip_object, "stop", Decoder, Stop, 0) != 0 {
    sentencepiece_unsupported(Decoder, DecoderKind)
  }
}

///|
fn validate_sentencepiece_pipeline(
  root : Map[String, Json],
) -> TokenizerJsonProfile raise TokenizerJsonError {
  validate_sentencepiece_normalizer(
    require_field(root, "normalizer", Root, NormalizerField),
  )
  if !(require_field(root, "pre_tokenizer", Root, PreTokenizerField) is Null) {
    sentencepiece_unsupported(PreTokenizer, PreTokenizerKind)
  }
  let bos_token_id = validate_sentencepiece_post_processor(
    require_field(root, "post_processor", Root, PostProcessorField),
  )
  validate_sentencepiece_decoder(
    require_field(root, "decoder", Root, DecoderField),
  )
  SentencePieceJson(bos_token_id~)
}
