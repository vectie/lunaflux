///|
const DIGIT_TRIPLET_SPLIT_REGEX : String = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

///|
fn validate_digit_triplet_pipeline(
  root : Map[String, Json],
) -> TokenizerJsonProfile raise TokenizerJsonError {
  let sequence = require_component_object(
    require_field(root, "pre_tokenizer", Root, PreTokenizerField),
    PreTokenizer,
  )
  validate_fields(sequence, PreTokenizer, ["type", "pretokenizers"])
  guard require_field(sequence, "pretokenizers", PreTokenizer, Components)
    is Array([split, bytelevel]) else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  let split = require_component_object(split, PreTokenizer)
  validate_fields(split, PreTokenizer, ["type", "pattern", "behavior", "invert"])
  let pattern = require_component_object(
    require_field(split, "pattern", PreTokenizer, Pattern),
    PreTokenizer,
  )
  validate_fields(pattern, PreTokenizer, ["Regex"])
  if require_string(split, "type", PreTokenizer, TypeField, 0) != "Split" ||
    require_string(split, "behavior", PreTokenizer, Behavior, 0) != "Isolated" ||
    require_bool(split, "invert", PreTokenizer, Invert, 0) ||
    require_string(pattern, "Regex", PreTokenizer, Pattern, 0) !=
    DIGIT_TRIPLET_SPLIT_REGEX {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  let (prefix, regex) = validate_bytelevel_component(bytelevel, PreTokenizer)
  if prefix || regex {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
  // These ByteLevel options affect offsets, not token IDs, in postprocessing
  // and decoding. Prefix-space insertion is disallowed in the pretokenizer.
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
  DigitTripletJson
}

///|
fn append_external_added_tokens(
  entries : Array[@tokenizer.VocabularyEntry],
  raw_ids : Map[String, Int],
  specials : Map[Int, AddedSpecial],
) -> Unit {
  let external : Array[AddedSpecial] = []
  for _, entry in specials {
    if !raw_ids.contains(entry.content) {
      external.push(entry)
    }
  }
  external.sort_by((left, right) => left.id.compare(right.id))
  for entry in external {
    entries.push({
      token_id: entry.id,
      piece: entry.piece,
      kind: if entry.special {
        Special
      } else {
        Added
      },
    })
  }
}
