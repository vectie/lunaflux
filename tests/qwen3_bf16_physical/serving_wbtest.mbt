///|
fn qwen3_serving_fixture(
  greedy? : String = "{\"input_ids\":[151643,9707],\"expected_token_ids\":[198,785]}",
) -> Bytes {
  @utf8.encode(
    "{\"schema\":\"lunaflux-qwen3-reference-corpus.v1\",\"greedy\":" +
    greedy +
    "}",
  )
}

///|
test "Qwen serving corpus binds exact prompt and complete greedy sequence" {
  let corpus = qwen3_greedy_corpus_case(qwen3_serving_fixture())
  assert_eq(corpus.input_ids, [151643, 9707])
  assert_eq(corpus.expected_ids, [198, 785])
  let limits = qwen3_serving_limits(corpus, 151935, 40960)
  let frame = qwen3_serving_frame(
    corpus,
    "a".repeat(64),
    "b".repeat(64),
    limits,
  ) catch {
    InvalidInput(message) => fail(message)
    CorpusMismatch(message) => fail(message)
    _ => fail("unexpected serving-frame error")
  }
  assert_true(frame.length() > 16)
  assert_eq(qwen3_token_sequence(corpus.expected_ids), "198,785")
}

///|
test "Qwen serving corpus rejects missing and empty greedy authority" {
  for
    hostile in [
      @utf8.encode("{\"schema\":\"lunaflux-qwen3-reference-corpus.v1\"}"),
      qwen3_serving_fixture(
        greedy="{\"input_ids\":[],\"expected_token_ids\":[1]}",
      ),
      qwen3_serving_fixture(
        greedy="{\"input_ids\":[1],\"expected_token_ids\":[]}",
      ),
    ] {
    try qwen3_greedy_corpus_case(hostile) catch {
      CorpusMismatch(_) | InvalidInput(_) => ()
      _ => fail("unexpected hostile serving-corpus error")
    } noraise {
      _ => fail("hostile serving corpus was admitted")
    }
  }
}
