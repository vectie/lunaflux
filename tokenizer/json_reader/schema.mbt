///|
priv struct AddedSpecial {
  id : Int
  content : String
  piece : Bytes
  source_index : Int
}

///|
fn parse_added_tokens(
  root : Map[String, Json],
  limits : @tokenizer.TokenizerLimits,
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
    if !special {
      raise UnsupportedSemantics(
        component=AddedToken,
        feature=OrdinaryAddedToken,
      )
    }
    if single_word || left_strip || right_strip || normalized {
      raise UnsupportedSemantics(
        component=AddedToken,
        feature=AddedTokenMatching,
      )
    }
    if id < 0 || by_id.contains(id) || content_ids.contains(content) {
      raise InvalidField(component=AddedToken, field=Id, index~)
    }
    let piece = @utf8.encode(content)
    by_id[id] = { id, content, piece, source_index: index }
    content_ids[content] = id
  }
  by_id
}

///|
fn validate_root_semantics(
  root : Map[String, Json],
) -> Unit raise TokenizerJsonError {
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
  require_null(
    root,
    "normalizer",
    Normalizer,
    NormalizerField,
    NormalizationBehavior,
  )
  validate_pre_tokenizer(
    require_field(root, "pre_tokenizer", Root, PreTokenizerField),
  )
  validate_post_processor(
    require_field(root, "post_processor", Root, PostProcessorField),
  )
  validate_decoder(require_field(root, "decoder", Root, DecoderField))
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
