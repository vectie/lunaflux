///|
fn digest() -> String {
  "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}

///|
test "model accepts canonical identity and preserves allowlist order" {
  let config = @model.new("/models/reference", digest(), [
    "LlamaForCausalLM", "DenseDecoder",
  ])
  assert_eq(config.artifact_path(), "/models/reference")
  assert_eq(config.expected_digest(), digest())
  assert_eq(config.architecture_allowlist()[0], "LlamaForCausalLM")
  assert_eq(config.architecture_allowlist()[1], "DenseDecoder")
}

///|
test "model rejects malformed digest and duplicate architecture" {
  try
    @model.new("/models/reference", "sha256:ABC", ["LlamaForCausalLM"])
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ExpectedDigest, issue=InvalidFormat, index=0)",
      )
  } noraise {
    _ => fail("expected digest rejection")
  }
  try
    @model.new("/models/reference", digest(), [
      "LlamaForCausalLM", "LlamaForCausalLM",
    ])
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ArchitectureName, issue=Duplicate, index=1)",
      )
  } noraise {
    _ => fail("expected duplicate architecture rejection")
  }
}

///|
test "model architecture count is bounded" {
  try
    @model.new("/models/reference", digest(), [
      "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12",
      "A13", "A14", "A15", "A16",
    ])
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ArchitectureAllowlist, issue=TooMany, index=16)",
      )
  } noraise {
    _ => fail("expected allowlist bound rejection")
  }
}
