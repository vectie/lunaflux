///|
fn validate_ordered_split(
  value : Json,
  regex : String,
) -> Unit raise TokenizerJsonError {
  let object = require_component_object(value, PreTokenizer)
  validate_fields(object, PreTokenizer, [
    "type", "pattern", "behavior", "invert",
  ])
  let pattern = require_component_object(
    require_field(object, "pattern", PreTokenizer, Pattern),
    PreTokenizer,
  )
  validate_fields(pattern, PreTokenizer, ["Regex"])
  guard require_string(object, "type", PreTokenizer, TypeField, 0) == "Split" &&
    require_string(object, "behavior", PreTokenizer, Behavior, 0) == "Isolated" &&
    !require_bool(object, "invert", PreTokenizer, Invert, 0) &&
    require_string(pattern, "Regex", PreTokenizer, Pattern, 0) == regex else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
}

///|
fn validate_ordered_script_pipeline(
  root : Map[String, Json],
) -> TokenizerJsonProfile raise TokenizerJsonError {
  let sequence = require_component_object(
    require_field(root, "pre_tokenizer", Root, PreTokenizerField),
    PreTokenizer,
  )
  validate_fields(sequence, PreTokenizer, ["type", "pretokenizers"])
  guard require_string(sequence, "type", PreTokenizer, TypeField, 0) ==
    "Sequence" &&
    require_field(sequence, "pretokenizers", PreTokenizer, Components)
    is Array([number, script, word, bytes]) else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  validate_ordered_split(number, ORDERED_NUMBER_REGEX)
  validate_ordered_split(script, ORDERED_SCRIPT_REGEX)
  validate_ordered_split(word, ORDERED_WORD_REGEX)
  let (prefix, regex) = validate_bytelevel_component(bytes, PreTokenizer)
  guard !prefix && !regex else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  match require_field(root, "post_processor", Root, PostProcessorField) {
    Null => ()
    value => ignore(validate_bytelevel_component(value, PostProcessor))
  }
  ignore(
    validate_bytelevel_component(
      require_field(root, "decoder", Root, DecoderField),
      Decoder,
    ),
  )
  OrderedScriptJson
}
