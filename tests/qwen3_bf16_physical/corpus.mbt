///|
fn qwen3_corpus_int(value : Json) -> Int raise Qwen3PhysicalError {
  guard value is Number(number, repr=_) else {
    raise CorpusMismatch("token ID is not numeric")
  }
  let integer = number.to_int()
  qwen3_require(integer.to_double() == number, "token ID is not exact")
  integer
}

///|
fn qwen3_corpus_ids(value : Json) -> Array[Int] raise Qwen3PhysicalError {
  guard value is Array(values) else {
    raise CorpusMismatch("token IDs are not an array")
  }
  values.map(qwen3_corpus_int)
}

///|
async fn qwen3_verify_reference_corpus(
  model_root : String,
  tokenizer_sha256 : String,
  corpus_path : String,
  corpus_sha256 : String,
) -> Int raise Qwen3PhysicalError {
  let tokenizer_bytes = qwen3_read_pinned(
    model_root + "/tokenizer.json",
    tokenizer_sha256,
    67108864,
  )
  let corpus_bytes = qwen3_read_pinned(corpus_path, corpus_sha256, 1048576)
  let limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: 65536,
    max_output_tokens: 65536,
    max_decoded_bytes: 1048576,
    max_vocab_entries: 200000,
    max_merge_rules: 1000000,
    max_token_bytes: 4096,
    max_special_tokens: 64,
  }
  let tokenizer = @json_reader.parse_bytes(
    tokenizer_bytes,
    digest=@tokenizer.TokenizerDigest::parse(tokenizer_sha256),
    limits~,
    max_bytes=67108864,
    max_depth=64,
  ) catch {
    _ => raise CorpusMismatch("Qwen3 tokenizer admission failed")
  }
  let parsed = @json.parse(@utf8.decode(corpus_bytes)) catch {
    _ => raise CorpusMismatch("reference corpus JSON is malformed")
  }
  guard parsed is Object(root) &&
    root["schema"] is String("lunaflux-qwen3-reference-corpus.v1") &&
    root["tokenizer_cases"] is Array(cases) &&
    root["chat"] is Object(chat) else {
    raise CorpusMismatch("reference corpus closed schema is absent")
  }
  qwen3_require(cases.length() >= 7, "reference corpus is incomplete")
  for case in cases {
    guard case is Object(fields) &&
      fields["text"] is String(text) &&
      fields["decoded"] is String(decoded) else {
      raise CorpusMismatch("tokenizer case is malformed")
    }
    let expected = qwen3_corpus_ids(fields["token_ids"])
    let actual = tokenizer.encode_text(
      text,
      special_policy=Allow,
      output_limit=65536,
      truncation=RejectOverflow,
    ) catch {
      _ => raise CorpusMismatch("tokenizer case encoding failed")
    }
    qwen3_require(actual.token_ids == expected, "tokenizer corpus IDs differ")
    let roundtrip = tokenizer.decode_text(expected, special_policy=Preserve) catch {
      _ => raise CorpusMismatch("tokenizer case decoding failed")
    }
    qwen3_require(roundtrip == decoded, "tokenizer corpus decode differs")
  }
  guard chat["rendered"] is String(rendered) else {
    raise CorpusMismatch("chat rendering is absent")
  }
  let chat_expected = qwen3_corpus_ids(chat["input_ids"])
  let chat_actual = tokenizer.encode_text(
    rendered,
    special_policy=Allow,
    output_limit=65536,
    truncation=RejectOverflow,
  ) catch {
    _ => raise CorpusMismatch("chat encoding failed")
  }
  qwen3_require(chat_actual.token_ids == chat_expected, "chat IDs differ")
  cases.length() + 1
}
