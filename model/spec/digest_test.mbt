///|
test "digest identities are constructed through checked public authority" {
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  let identity = @spec.ModelIdentity::new(content, plan)
  assert_eq(identity.content(), content)
  assert_eq(identity.plan(), plan)
}

///|
test "content and plan digest constructors reject noncanonical SHA-256 text" {
  for
    value in [
      "", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde", "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
      "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
    try @spec.ContentDigest::from_sha256(value) catch {
      InvalidDigest => ()
      _ => fail("unexpected content-digest error")
    } noraise {
      _ => fail("expected invalid content digest")
    }
    try @spec.PlanDigest::from_sha256(value) catch {
      InvalidDigest => ()
      _ => fail("unexpected plan-digest error")
    } noraise {
      _ => fail("expected invalid plan digest")
    }
  }
}
