///|
test "host manifest retains supplied numeric-binding label without hashing layout" {
  let content = @spec.ContentDigest::from_sha256(String::make(64, 'a'))
  let plan = @spec.PlanDigest::from_sha256(String::make(64, 'b'))
  let binding = @spec.PlanDigest::from_sha256(String::make(64, 'c'))
  let expected = DeepSeekV4HostExpectedDigests::new(
    identity=@spec.ModelIdentity::new(content, plan),
    logical_manifest_digest=plan,
    numeric_schema_digest=plan,
    numeric_binding_digest=binding,
  )
  assert_eq(manifest_source_label(expected).as_hex(), binding.as_hex())
  let other = { ..expected, numeric_binding_digest: plan, }
  assert_eq(manifest_source_label(other).as_hex(), plan.as_hex())
  assert_true(manifest_source_label(expected) != manifest_source_label(other))
}
