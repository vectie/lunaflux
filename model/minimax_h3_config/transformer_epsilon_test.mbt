///|
test "original transformer accepts installed separate operation epsilons" {
  let source = original_transformer_config().replace(
    old="\"eps\":1e-5",
    new="\"norm_eps\":1e-5,\"qk_norm_eps\":1e-5,\"final_norm_eps\":1e-5",
  )
  let metadata = @minimax_h3_config.parse_transformer(
    source,
    variant=Fl2va,
    content_digest=h3_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(metadata.spec().checkpoint_format(), OriginalV1)
  assert_eq(metadata.spec().layer_count(), 50)
  // Both spellings lower to the same semantic normalization contract.
  let shared = @minimax_h3_config.parse_transformer(
    original_transformer_config(),
    variant=Fl2va,
    content_digest=h3_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(metadata.spec(), shared.spec())
}

///|
test "original transformer rejects partial ambiguous and shifted epsilons" {
  for
    fields in [
      "\"norm_eps\":1e-5", "\"norm_eps\":1e-5,\"qk_norm_eps\":1e-5", "\"eps\":1e-5,\"norm_eps\":1e-5",
      "\"eps\":1e-5,\"norm_eps\":1e-4,\"qk_norm_eps\":1e-5,\"final_norm_eps\":1e-5",
      "\"norm_eps\":1e-5,\"qk_norm_eps\":1e-4,\"final_norm_eps\":1e-5", "\"norm_eps\":null,\"qk_norm_eps\":1e-5,\"final_norm_eps\":1e-5",
    ] {
    try
      @minimax_h3_config.parse_transformer(
        original_transformer_config().replace(old="\"eps\":1e-5", new=fields),
        variant=Fl2va,
        content_digest=h3_digest(),
        max_bytes=4096,
        max_depth=8,
      )
    catch {
      MissingField | InvalidField => ()
      _ => fail("unexpected epsilon dialect error")
    } noraise {
      _ => fail("invalid epsilon dialect admitted")
    }
  }
}
