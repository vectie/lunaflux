///|
priv struct AddedSpecial {
  id : Int
  content : String
  piece : Bytes
  special : Bool
  source_index : Int
}

///|
priv enum TokenizerJsonProfile {
  ByteLevelJson
  SentencePieceJson(bos_token_id~ : Int)
  Qwen3Json
}

///|
const QWEN3_FIRST_ADDED_TOKEN_ID : Int = 151643

///|
const QWEN3_LAST_ADDED_TOKEN_ID : Int = 151668

///|
const QWEN3_SPLIT_REGEX : String = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

///|
fn qwen3_added_token_content(id : Int) -> String? {
  match id {
    151643 => Some("<|endoftext|>")
    151644 => Some("<|im_start|>")
    151645 => Some("<|im_end|>")
    151646 => Some("<|object_ref_start|>")
    151647 => Some("<|object_ref_end|>")
    151648 => Some("<|box_start|>")
    151649 => Some("<|box_end|>")
    151650 => Some("<|quad_start|>")
    151651 => Some("<|quad_end|>")
    151652 => Some("<|vision_start|>")
    151653 => Some("<|vision_end|>")
    151654 => Some("<|vision_pad|>")
    151655 => Some("<|image_pad|>")
    151656 => Some("<|video_pad|>")
    151657 => Some("<tool_call>")
    151658 => Some("</tool_call>")
    151659 => Some("<|fim_prefix|>")
    151660 => Some("<|fim_middle|>")
    151661 => Some("<|fim_suffix|>")
    151662 => Some("<|fim_pad|>")
    151663 => Some("<|repo_name|>")
    151664 => Some("<|file_sep|>")
    151665 => Some("<tool_response>")
    151666 => Some("</tool_response>")
    151667 => Some("<think>")
    151668 => Some("</think>")
    _ => None
  }
}

///|
fn parse_added_tokens(
  root : Map[String, Json],
  limits : @tokenizer.TokenizerLimits,
  profile : TokenizerJsonProfile,
) -> Map[Int, AddedSpecial] raise TokenizerJsonError {
  guard require_field(root, "added_tokens", Root, AddedTokens) is Array(entries) else {
    raise InvalidField(component=Root, field=AddedTokens, index=0)
  }
  if entries.length() > limits.max_special_tokens {
    raise AddedTokenLimitExceeded(limit=limits.max_special_tokens)
  }
  let by_id : Map[Int, AddedSpecial] = Map([], capacity=entries.length())
  let content_ids : Map[String, Int] = Map([], capacity=entries.length())
  for index, value in entries {
    guard value is Object(entry) else {
      raise InvalidField(component=AddedToken, field=AddedTokens, index~)
    }
    validate_fields(entry, AddedToken, [
      "id", "content", "single_word", "lstrip", "rstrip", "normalized", "special",
    ])
    let id = require_exact_int(entry, "id", AddedToken, Id, index)
    let content = require_string(entry, "content", AddedToken, Content, index)
    let single_word = require_bool(
      entry,
      "single_word",
      AddedToken,
      SingleWord,
      index,
    )
    let left_strip = require_bool(entry, "lstrip", AddedToken, LeftStrip, index)
    let right_strip = require_bool(
      entry,
      "rstrip",
      AddedToken,
      RightStrip,
      index,
    )
    let normalized = require_bool(
      entry,
      "normalized",
      AddedToken,
      Normalized,
      index,
    )
    let special = require_bool(entry, "special", AddedToken, Special, index)
    if single_word || left_strip || right_strip || normalized {
      raise UnsupportedSemantics(
        component=AddedToken,
        feature=AddedTokenMatching,
      )
    }
    if id < 0 || by_id.contains(id) || content_ids.contains(content) {
      raise InvalidField(component=AddedToken, field=Id, index~)
    }
    match profile {
      Qwen3Json => {
        guard id == QWEN3_FIRST_ADDED_TOKEN_ID + index &&
          qwen3_added_token_content(id) is Some(expected) &&
          content == expected &&
          special == (id <= 151656) else {
          raise UnsupportedSemantics(
            component=AddedToken,
            feature=Qwen3AddedTokens,
          )
        }
      }
      _ =>
        if !special {
          raise UnsupportedSemantics(
            component=AddedToken,
            feature=OrdinaryAddedToken,
          )
        }
    }
    let piece = @utf8.encode(content)
    by_id[id] = { id, content, piece, special, source_index: index }
    content_ids[content] = id
  }
  if profile is Qwen3Json &&
    entries.length() !=
    QWEN3_LAST_ADDED_TOKEN_ID - QWEN3_FIRST_ADDED_TOKEN_ID + 1 {
    raise UnsupportedSemantics(component=AddedToken, feature=Qwen3AddedTokens)
  }
  by_id
}

///|
fn validate_root_semantics(
  root : Map[String, Json],
) -> TokenizerJsonProfile raise TokenizerJsonError {
  validate_fields(root, Root, [
    "version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer",
    "post_processor", "decoder", "model",
  ])
  let version = require_string(root, "version", Root, Version, 0)
  if version != "1.0" {
    raise UnsupportedSemantics(component=Root, feature=SchemaVersion)
  }
  require_null(root, "truncation", Root, Truncation, TruncationBehavior)
  require_null(root, "padding", Root, Padding, PaddingBehavior)
  match require_field(root, "normalizer", Root, NormalizerField) {
    Null => {
      validate_pre_tokenizer(
        require_field(root, "pre_tokenizer", Root, PreTokenizerField),
      )
      validate_post_processor(
        require_field(root, "post_processor", Root, PostProcessorField),
      )
      validate_decoder(require_field(root, "decoder", Root, DecoderField))
      ByteLevelJson
    }
    Object(normalizer) =>
      if normalizer.get("type") is Some(String("NFC")) {
        match require_field(root, "pre_tokenizer", Root, PreTokenizerField) {
          Object(pretokenizer) if pretokenizer.get("type")
            is Some(String("Sequence")) => {
            validate_qwen3_pipeline(root)
            Qwen3Json
          }
          _ =>
            raise UnsupportedSemantics(
              component=Normalizer,
              feature=NormalizationBehavior,
            )
        }
      } else {
        validate_sentencepiece_pipeline(root)
      }
    _ => validate_sentencepiece_pipeline(root)
  }
}

///|
fn validate_qwen3_split(value : Json) -> Unit raise TokenizerJsonError {
  let object = require_component_object(value, PreTokenizer)
  validate_fields(object, PreTokenizer, [
    "type", "pattern", "behavior", "invert",
  ])
  if require_string(object, "type", PreTokenizer, TypeField, 0) != "Split" ||
    require_string(object, "behavior", PreTokenizer, Behavior, 0) != "Isolated" ||
    require_bool(object, "invert", PreTokenizer, Invert, 0) {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  let pattern = require_component_object(
    require_field(object, "pattern", PreTokenizer, Pattern),
    PreTokenizer,
  )
  validate_fields(pattern, PreTokenizer, ["Regex"])
  if require_string(pattern, "Regex", PreTokenizer, Pattern, 0) !=
    QWEN3_SPLIT_REGEX {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
}

///|
fn validate_qwen3_pipeline(
  root : Map[String, Json],
) -> Unit raise TokenizerJsonError {
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
  if require_string(sequence, "type", PreTokenizer, TypeField, 0) != "Sequence" {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  guard require_field(sequence, "pretokenizers", PreTokenizer, Components)
    is Array([split, bytelevel]) else {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  validate_qwen3_split(split)
  let (add_prefix, use_regex) = validate_bytelevel_component(
    bytelevel,
    PreTokenizer,
  )
  let bytelevel_object = require_component_object(bytelevel, PreTokenizer)
  if add_prefix ||
    use_regex ||
    require_bool(bytelevel_object, "trim_offsets", PreTokenizer, TrimOffsets, 0) {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=Qwen3PreTokenization,
    )
  }
  let post_value = require_field(
    root,
    "post_processor",
    Root,
    PostProcessorField,
  )
  let post = require_component_object(post_value, PostProcessor)
  let (post_prefix, post_regex) = validate_bytelevel_component(
    post_value,
    PostProcessor,
  )
  let decoder_value = require_field(root, "decoder", Root, DecoderField)
  let decoder = require_component_object(decoder_value, Decoder)
  let (decoder_prefix, decoder_regex) = validate_bytelevel_component(
    decoder_value,
    Decoder,
  )
  if post_prefix ||
    post_regex ||
    decoder_prefix ||
    decoder_regex ||
    require_bool(post, "trim_offsets", PostProcessor, TrimOffsets, 0) ||
    require_bool(decoder, "trim_offsets", Decoder, TrimOffsets, 0) {
    raise UnsupportedSemantics(
      component=PostProcessor,
      feature=Qwen3PreTokenization,
    )
  }
}

///|
fn validate_bytelevel_component(
  value : Json,
  component : JsonComponent,
) -> (Bool, Bool) raise TokenizerJsonError {
  guard value is Object(object) else {
    raise InvalidField(component~, field=TypeField, index=0)
  }
  validate_fields(object, component, [
    "type", "add_prefix_space", "trim_offsets", "use_regex",
  ])
  let kind = require_string(object, "type", component, TypeField, 0)
  if kind != "ByteLevel" {
    let feature = match component {
      Decoder => DecoderKind
      PostProcessor => PostProcessingBehavior
      _ => PreTokenizerKind
    }
    raise UnsupportedSemantics(component~, feature~)
  }
  let add_prefix = require_bool(
    object,
    "add_prefix_space",
    component,
    AddPrefixSpace,
    0,
  )
  ignore(require_bool(object, "trim_offsets", component, TrimOffsets, 0))
  let use_regex = require_bool(object, "use_regex", component, UseRegex, 0)
  (add_prefix, use_regex)
}

///|
fn validate_pre_tokenizer(value : Json) -> Unit raise TokenizerJsonError {
  let (add_prefix, use_regex) = validate_bytelevel_component(
    value,
    PreTokenizer,
  )
  if add_prefix {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=PrefixSpaceInsertion,
    )
  }
  if use_regex {
    raise UnsupportedSemantics(
      component=PreTokenizer,
      feature=RegexPreTokenization,
    )
  }
}

///|
fn validate_post_processor(value : Json) -> Unit raise TokenizerJsonError {
  match value {
    Null => ()
    _ => {
      let (add_prefix, use_regex) = validate_bytelevel_component(
        value,
        PostProcessor,
      )
      if add_prefix || use_regex {
        raise UnsupportedSemantics(
          component=PostProcessor,
          feature=PostProcessingBehavior,
        )
      }
    }
  }
}

///|
fn validate_decoder(value : Json) -> Unit raise TokenizerJsonError {
  let (add_prefix, use_regex) = validate_bytelevel_component(value, Decoder)
  if add_prefix || use_regex {
    raise UnsupportedSemantics(component=Decoder, feature=DecoderKind)
  }
}
