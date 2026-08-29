///|
priv struct Qwen3GreedyCorpusCase {
  input_ids : Array[Int]
  expected_ids : Array[Int]
}

///|
fn qwen3_greedy_corpus_case(
  bytes : BytesView,
) -> Qwen3GreedyCorpusCase raise Qwen3PhysicalError {
  let parsed = @json.parse(@utf8.decode(bytes)) catch {
    _ => raise CorpusMismatch("reference corpus JSON is malformed")
  }
  guard parsed is Object(root) &&
    root.get("schema") is Some(String("lunaflux-qwen3-reference-corpus.v1")) &&
    root.get("greedy") is Some(Object(greedy)) &&
    greedy.get("input_ids") is Some(input_value) &&
    greedy.get("expected_token_ids") is Some(expected_value) else {
    raise CorpusMismatch("reference corpus greedy case is absent")
  }
  let input_ids = qwen3_corpus_ids(input_value)
  let expected_ids = qwen3_corpus_ids(expected_value)
  qwen3_require(
    !input_ids.is_empty() && input_ids.length() <= 4096,
    "greedy input IDs are outside bounds",
  )
  qwen3_require(
    !expected_ids.is_empty() && expected_ids.length() <= 256,
    "greedy expected IDs are outside bounds",
  )
  { input_ids, expected_ids }
}
