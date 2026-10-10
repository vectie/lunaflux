///|
test "dependent chain source consumes distinct previous tokens and pre-normalized hidden" {
  for (temperature, prefix) in [(0.0, "chain_greedy"), (0.7, "chain_random")] {
    let source = @sequential_markov_cuda_source.sequential_markov_source(
      @precision.SequentialMarkovPrecision::new(
        vocabulary_size=17,
        hidden_width=8,
        embedding_width=4,
        maximum_rows=5,
        temperature~,
        maximum_workspace_bytes=60L,
      ),
      symbol_prefix=prefix,
    )
    let text = @utf8.decode(source.source_bytes())
    assert_true(text.contains("tokens[row+1]"))
    assert_true(text.contains("const int token=tokens[row]"))
    assert_false(text.contains("sigmoid"))
    assert_true(text.contains("hidden[(uint64_t)row*8+c]"))
    if temperature > 0.0 {
      assert_true(text.contains("_draw(value,rng[0],rng[1],row,word)"))
    }
    if @env.get_env_var("LUNA_EXPORT_MARKOV") == Some("1") {
      println("LUNA_MARKOV_BEGIN")
      println(text)
      println("LUNA_MARKOV_END")
    }
  }
  if @env.get_env_var("LUNA_EXPORT_MARKOV") == Some("1") {
    println("LUNA_MARKOV_BEGIN")
    println(
      @utf8.decode(
        @text_source.seeded_token_block_source(
          @precision.SeededTokenBlockPrecision::new(
            vocabulary_size=17,
            block_rows=5,
            maximum_main_rows=8,
            noise_token_id=16,
          ),
          symbol_prefix="seed_fixture",
        ),
      ),
    )
    println("LUNA_MARKOV_END")
  }
}
