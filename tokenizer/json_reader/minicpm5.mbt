///|
const MINICPM5_WORD_REGEX : String = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

///|
fn validate_minicpm5_split(
  value : Json,
  regex : String,
) -> Unit raise TokenizerJsonError {
  let split = require_component_object(value, PreTokenizer)
  validate_fields(split, PreTokenizer, ["type", "pattern", "behavior", "invert"])
  let pattern = require_component_object(
    require_field(split, "pattern", PreTokenizer, Pattern),
    PreTokenizer,
  )
  validate_fields(pattern, PreTokenizer, ["Regex"])
  if require_string(split, "type", PreTokenizer, TypeField, 0) != "Split" ||
    require_string(split, "behavior", PreTokenizer, Behavior, 0) != "Isolated" ||
    require_bool(split, "invert", PreTokenizer, Invert, 0) ||
    require_string(pattern, "Regex", PreTokenizer, Pattern, 0) != regex {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
}

///|
fn validate_minicpm5_pipeline(
  root : Map[String, Json],
) -> TokenizerJsonProfile raise TokenizerJsonError {
  let sequence = require_component_object(root["pre_tokenizer"], PreTokenizer)
  validate_fields(sequence, PreTokenizer, ["type", "pretokenizers"])
  guard sequence["pretokenizers"] is Array([numbers, words, bytes]) else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  validate_minicpm5_split(numbers, "\\p{N}{1,3}")
  validate_minicpm5_split(words, MINICPM5_WORD_REGEX)
  let (prefix, regex) = validate_bytelevel_component(bytes, PreTokenizer)
  if prefix || regex {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  ignore(validate_bytelevel_component(root["decoder"], Decoder))
  let bos = validate_sentencepiece_post_processor(
    root["post_processor"],
    expected_bos=0,
  )
  if bos != 0 {
    raise UnsupportedSemantics(
      component=PostProcessor,
      feature=PostProcessingBehavior,
    )
  }
  MiniCPM5Json
}
