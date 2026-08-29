///|
fn qwen35_added_tokens() -> Array[Json] {
  let entries : Array[Json] = []
  for id in QWEN35_FIRST_ADDED_TOKEN_ID..<=QWEN35_LAST_ADDED_TOKEN_ID {
    entries.push(
      Json::object({
        "id": Json::number(id.to_double()),
        "content": Json::string(
          qwen_added_token_content(id - QWEN35_FIRST_ADDED_TOKEN_ID).unwrap(),
        ),
        "single_word": Json::boolean(false),
        "lstrip": Json::boolean(false),
        "rstrip": Json::boolean(false),
        "normalized": Json::boolean(false),
        "special": Json::boolean(id <= QWEN35_FIRST_ADDED_TOKEN_ID + 13),
      }),
    )
  }
  entries
}

///|
test "exact Qwen3.5 and Qwen3.6 tokenizer metadata selects its own profile" {
  let root = qwen3_test_root(regex=QWEN35_SPLIT_REGEX)
  guard validate_root_semantics(root) is Qwen35Json else {
    fail("expected separate Qwen3.5/Qwen3.6 profile")
  }
  root["added_tokens"] = Json::array(qwen35_added_tokens())
  let added = parse_added_tokens(root, qwen3_test_limits(), Qwen35Json)
  assert_eq(QWEN35_FIRST_ADDED_TOKEN_ID, 248044)
  assert_eq(QWEN35_LAST_ADDED_TOKEN_ID, 248069)
  assert_eq(added.length(), 26)
  assert_eq(added[248044].content, "<|endoftext|>")
  assert_eq(added[248045].content, "<|im_start|>")
  assert_eq(added[248069].content, "</think>")
  assert_true(added[248057].special)
  assert_false(added[248058].special)
  assert_false(added[248069].special)
}

///|
test "Qwen3.5 profile pins ordinary vocabulary boundary and mark regex" {
  assert_eq(qwen_profile_first_added(Qwen35Json), 248044)
  assert_eq(qwen_profile_last_added(Qwen35Json), 248069)
  guard validate_root_semantics(qwen3_test_root()) is Qwen3Json else {
    fail("dense Qwen3 regex changed profile")
  }
  guard validate_root_semantics(qwen3_test_root(regex=QWEN35_SPLIT_REGEX))
    is Qwen35Json else {
    fail("mark-aware regex did not select Qwen3.5")
  }
  try
    validate_root_semantics(
      qwen3_test_root(
        regex="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      ),
    )
  catch {
    UnsupportedSemantics(component=PreTokenizer, feature=Qwen3PreTokenization) =>
      ()
    _ => fail("unexpected changed-regex error")
  } noraise {
    _ => fail("partially mark-aware regex was admitted")
  }
}

///|
test "Qwen3.5 added-token IDs contents and kinds fail closed" {
  let root = qwen3_test_root(regex=QWEN35_SPLIT_REGEX)
  let wrong_id = qwen35_added_tokens()
  guard wrong_id[0] is Object(first) else { fail("fixture invariant") }
  first["id"] = Json::number(248043.0)
  root["added_tokens"] = Json::array(wrong_id)
  try parse_added_tokens(root, qwen3_test_limits(), Qwen35Json) catch {
    UnsupportedSemantics(component=AddedToken, feature=Qwen3AddedTokens) => ()
    _ => fail("unexpected Qwen3.5 ID error")
  } noraise {
    _ => fail("shifted Qwen3.5 token IDs were admitted")
  }
  let wrong_kind = qwen35_added_tokens()
  guard wrong_kind[25] is Object(last) else { fail("fixture invariant") }
  last["special"] = Json::boolean(true)
  root["added_tokens"] = Json::array(wrong_kind)
  try parse_added_tokens(root, qwen3_test_limits(), Qwen35Json) catch {
    UnsupportedSemantics(component=AddedToken, feature=Qwen3AddedTokens) => ()
    _ => fail("unexpected Qwen3.5 kind error")
  } noraise {
    _ => fail("changed Qwen3.5 added-token kind was admitted")
  }
}
