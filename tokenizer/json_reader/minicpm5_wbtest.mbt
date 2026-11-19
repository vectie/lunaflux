///|
fn minicpm5_root() -> Map[String, Json] raise {
  let root = qwen3_test_root(regex=MINICPM5_WORD_REGEX)
  root["normalizer"] = Json::null()
  guard root["pre_tokenizer"] is { "pretokenizers": Array(parts), .. } else {
    fail("fixture")
  }
  parts.insert(
    0,
    Json::object({
      "type": Json::string("Split"),
      "pattern": Json::object({ "Regex": Json::string("\\p{N}{1,3}") }),
      "behavior": Json::string("Isolated"),
      "invert": Json::boolean(false),
    }),
  )
  let post_processor =
    #|{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<s>","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<s>":{"id":"<s>","ids":[0],"tokens":["<s>"]}}}
  root["post_processor"] = @json.parse(post_processor)
  root
}

///|
test "MiniCPM5 requires ordered splits and BOS zero rather than SentencePiece BOS one" {
  let root = minicpm5_root()
  assert_true(validate_root_semantics(root) is MiniCPM5Json)
  guard root["pre_tokenizer"] is { "pretokenizers": Array(parts), .. } else {
    fail("fixture")
  }
  let first = parts[0]
  parts[0] = parts[1]
  parts[1] = first
  assert_true(
    try validate_root_semantics(root) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let root = minicpm5_root()
  guard root["post_processor"]
    is { "special_tokens": { "<s>": Object(bos), .. }, .. } else {
    fail("fixture")
  }
  bos["ids"] = Json::array([Json::number(1)])
  assert_true(
    try validate_root_semantics(root) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
