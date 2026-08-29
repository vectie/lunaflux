///|
test "Mistral configuration admits exact semantics and caps full attention" {
  let document = selected_mistral_config()
  let from_text = parse_selected_mistral(document)
  let from_bytes = @config_reader.parse_mistral_bytes(
    @utf8.encode(document),
    content_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(from_text, from_bytes)
  let spec = from_text.spec()
  assert_eq(spec.vocabulary_size(), 32000)
  assert_eq(spec.max_position_embeddings(), 128)
  assert_eq(spec.sliding_window(), Some(64))
  assert_eq(spec.context_length(), 64)
  assert_eq(spec.head_dimension(), 4)
}

///|
test "Mistral parser does not widen the selected Llama parser" {
  try parse_selected(selected_mistral_config()) catch {
    InvalidValue(field=Architectures, issue=UnsupportedValue, index=0) => ()
    _ => fail("unexpected Llama family rejection")
  } noraise {
    _ => fail("Llama parser admitted Mistral")
  }
  try parse_selected_mistral(selected_config()) catch {
    InvalidValue(field=Architectures, issue=UnsupportedValue, index=0) => ()
    _ => fail("unexpected Mistral family rejection")
  } noraise {
    _ => fail("Mistral parser admitted Llama")
  }
}

///|
test "Mistral rejects unsafe sliding-window and family substitutions" {
  for invalid_window in ["0", "-1"] {
    try
      parse_selected_mistral(
        selected_mistral_config().replace(
          old="\"sliding_window\":64",
          new="\"sliding_window\":" + invalid_window,
        ),
      )
    catch {
      Validation(InvalidDimension(ContextLength)) => ()
      _ => fail("unexpected Mistral window rejection")
    } noraise {
      _ => fail("unsafe Mistral sliding window was admitted")
    }
  }
  try
    parse_selected_mistral(
      selected_mistral_config().replace(
        old="\"model_type\":\"mistral\"",
        new="\"model_type\":\"llama\"",
      ),
    )
  catch {
    InvalidValue(field=ModelType, issue=UnsupportedValue, index=0) => ()
    _ => fail("unexpected Mistral model-type rejection")
  } noraise {
    _ => fail("foreign Mistral model type was admitted")
  }
}

///|
test "Mistral null window retains the declared full-attention ceiling" {
  let metadata = parse_selected_mistral(
    selected_mistral_config().replace(
      old="\"sliding_window\":64",
      new="\"sliding_window\":null",
    ),
  )
  assert_eq(metadata.spec().sliding_window(), None)
  assert_eq(metadata.spec().context_length(), 128)
}
