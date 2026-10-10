///|
test "host manifest plan label is supplied metadata not a component checksum" {
  let identity = fixture_identity()
  assert_eq(manifest_source_label(identity).as_hex(), identity.plan().as_hex())
  let other = @spec.ModelIdentity::new(
    identity.content(),
    @spec.PlanDigest::from_sha256(String::make(64, '0')),
  )
  assert_eq(manifest_source_label(other).as_hex(), other.plan().as_hex())
  assert_true(manifest_source_label(identity) != manifest_source_label(other))
}
