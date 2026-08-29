///|
fn sentencepiece_hex_byte(value : Int) -> String {
  let hex = value.to_string(radix=16).to_upper()
  if hex.length() == 1 {
    "0\{hex}"
  } else {
    hex
  }
}

///|
fn sentencepiece_added_token(id : Int, content : String) -> Json {
  Json::object({
    "id": Json::number(id.to_double()),
    "content": Json::string(content),
    "single_word": Json::boolean(false),
    "lstrip": Json::boolean(false),
    "rstrip": Json::boolean(false),
    "normalized": Json::boolean(false),
    "special": Json::boolean(true),
  })
}

///|
fn sentencepiece_template_item(
  kind : String,
  id : String,
  type_id : Int,
) -> Json {
  let wrapper : Map[String, Json] = Map([])
  wrapper[kind] = Json::object({
    "id": Json::string(id),
    "type_id": Json::number(type_id.to_double()),
  })
  Json::object(wrapper)
}

///|
fn sentencepiece_document(
  normalized_content? : String = "▁",
  fuse_unknown? : Bool = true,
  byte_fallback? : Bool = true,
  decoder_start? : Int = 1,
  template_bos_id? : Int = 1,
) -> String {
  let vocab : Map[String, Json] = {
    "<unk>": Json::number(0.0),
    "<s>": Json::number(1.0),
    "</s>": Json::number(2.0),
  }
  for value in 0..<256 {
    vocab["<0x\{sentencepiece_hex_byte(value)}>"] = Json::number(
      (value + 3).to_double(),
    )
  }
  vocab["▁L"] = Json::number(259.0)
  Json::object({
    "version": Json::string("1.0"),
    "truncation": Json::null(),
    "padding": Json::null(),
    "added_tokens": Json::array([
      sentencepiece_added_token(0, "<unk>"),
      sentencepiece_added_token(1, "<s>"),
      sentencepiece_added_token(2, "</s>"),
    ]),
    "normalizer": Json::object({
      "type": Json::string("Sequence"),
      "normalizers": Json::array([
        Json::object({
          "type": Json::string("Prepend"),
          "prepend": Json::string("▁"),
        }),
        Json::object({
          "type": Json::string("Replace"),
          "pattern": Json::object({ "String": Json::string(" ") }),
          "content": Json::string(normalized_content),
        }),
      ]),
    }),
    "pre_tokenizer": Json::null(),
    "post_processor": Json::object({
      "type": Json::string("TemplateProcessing"),
      "single": Json::array([
        sentencepiece_template_item("SpecialToken", "<s>", 0),
        sentencepiece_template_item("Sequence", "A", 0),
      ]),
      "pair": Json::array([
        sentencepiece_template_item("SpecialToken", "<s>", 0),
        sentencepiece_template_item("Sequence", "A", 0),
        sentencepiece_template_item("SpecialToken", "<s>", 1),
        sentencepiece_template_item("Sequence", "B", 1),
      ]),
      "special_tokens": Json::object({
        "<s>": Json::object({
          "id": Json::string("<s>"),
          "ids": Json::array([Json::number(template_bos_id.to_double())]),
          "tokens": Json::array([Json::string("<s>")]),
        }),
      }),
    }),
    "decoder": Json::object({
      "type": Json::string("Sequence"),
      "decoders": Json::array([
        Json::object({
          "type": Json::string("Replace"),
          "pattern": Json::object({ "String": Json::string("▁") }),
          "content": Json::string(" "),
        }),
        Json::object({ "type": Json::string("ByteFallback") }),
        Json::object({ "type": Json::string("Fuse") }),
        Json::object({
          "type": Json::string("Strip"),
          "content": Json::string(" "),
          "start": Json::number(decoder_start.to_double()),
          "stop": Json::number(0.0),
        }),
      ]),
    }),
    "model": Json::object({
      "type": Json::string("BPE"),
      "dropout": Json::null(),
      "unk_token": Json::string("<unk>"),
      "continuing_subword_prefix": Json::null(),
      "end_of_word_suffix": Json::null(),
      "fuse_unk": Json::boolean(fuse_unknown),
      "byte_fallback": Json::boolean(byte_fallback),
      "vocab": Json::object(vocab),
      "merges": Json::array([]),
    }),
  }).stringify()
}

///|
fn parse_sentencepiece_document(
  document : String,
) -> @tokenizer.TokenizerSpec raise @json_reader.TokenizerJsonError {
  @json_reader.parse_string(
    document,
    digest=test_digest(),
    limits=test_limits(),
    max_bytes=65536,
    max_depth=16,
  )
}

///|
test "selected SentencePiece JSON profile is exact and executable" {
  let spec = parse_sentencepiece_document(sentencepiece_document())
  assert_eq(spec.vocab_size(), 260)
  assert_eq(spec.merge_count(), 0)
  assert_eq(
    spec.encode_text(
      "Hi",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    [1, 229, 153, 132, 75, 108],
  )
}

///|
test "nearby SentencePiece normalizer template decoder and model profiles fail closed" {
  let variants : Array[(String, JsonComponent, UnsupportedFeature)] = [
    (
      sentencepiece_document(normalized_content="_"),
      Normalizer,
      NormalizationBehavior,
    ),
    (
      sentencepiece_document(template_bos_id=2),
      PostProcessor,
      PostProcessingBehavior,
    ),
    (sentencepiece_document(decoder_start=0), Decoder, DecoderKind),
    (sentencepiece_document(fuse_unknown=false), Model, UnknownTokenFallback),
    (sentencepiece_document(byte_fallback=false), Model, UnknownTokenFallback),
  ]
  for variant in variants {
    let (document, component, feature) = variant
    try ignore(parse_sentencepiece_document(document)) catch {
      UnsupportedSemantics(component=actual_component, feature=actual_feature) => {
        assert_eq(actual_component, component)
        assert_eq(actual_feature, feature)
      }
      _ => fail("unexpected SentencePiece near-profile failure")
    } noraise {
      _ => fail("nearby SentencePiece profile was accepted")
    }
  }
}
