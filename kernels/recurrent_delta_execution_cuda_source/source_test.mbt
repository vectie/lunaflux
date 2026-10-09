///|
test "recurrent lowering separates batch sequences from persistent slots" {
  let plan = @precision.RecurrentDeltaPrecision::new(
    head_count=2,
    key_dimension=4,
    value_dimension=6,
    maximum_rows=4,
    maximum_sequences=2,
    cache_slots=4,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_state_bytes=768L,
  )
  let source = @recurrent_delta_execution_cuda_source.recurrent_delta_source(
    plan,
    symbol="delta",
  )
  assert_eq(source.launch(), (2, 2, 32))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("slot=slots[sequence]"))
  assert_true(text.contains("reset[sequence] ? 0.0f"))
  assert_true(text.contains("float memory_state[4]"))
}

///|
test "export recurrent execution fixtures" {
  if @env.get_env_var("LUNA_EXPORT_RECURRENT") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN recurrent")
  for
    (name, heads, keys, values, rows, sequences) in [
      ("delta_small", 2, 4, 6, 4, 2),
      ("delta_medium", 4, 32, 16, 8, 4),
      ("delta_glm", 64, 128, 128, 8, 2),
    ] {
    let plan = @precision.RecurrentDeltaPrecision::new(
      head_count=heads,
      key_dimension=keys,
      value_dimension=values,
      maximum_rows=rows,
      maximum_sequences=sequences,
      cache_slots=4,
      normalization_epsilon=0.000001,
      log_decay_lower_bound=-5.0,
      maximum_state_bytes=16777216L,
    )
    println(
      @utf8.decode(
        @recurrent_delta_execution_cuda_source.recurrent_delta_source(
          plan,
          symbol=name,
        ).source_bytes(),
      ),
    )
  }
  for
    (name, heads, keys, values, rows, sequences) in [
      ("delta_small", 2, 4, 6, 4, 2),
      ("delta_medium", 4, 32, 16, 8, 4),
      ("delta_glm", 64, 128, 128, 8, 2),
    ] {
    println(
      @utf8.decode(
        @legacy.causal_normalized_recurrent_delta_bf16_f32_source(
          head_count=heads,
          key_dimension=keys,
          value_dimension=values,
          maximum_rows=rows,
          maximum_sequences=sequences,
          normalization_epsilon=0.000001,
          log_decay_lower_bound=-5.0,
          function_symbol=name + "_legacy",
        ).source_bytes(),
      ),
    )
  }
  println("LUNA_SOURCE_END")
}
