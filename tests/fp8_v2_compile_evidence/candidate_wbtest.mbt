///|
test "compile candidates are deterministic, target-bound, and v2-only" {
  let digest = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
  let first = try! prepare_compile_candidates("sm_89", digest)
  let repeated = try! prepare_compile_candidates("sm_89", digest)
  let hopper = try! prepare_compile_candidates("sm_90", digest)
  let blackwell = try! prepare_compile_candidates("sm_120", digest)
  assert_eq(first.manifest_sha256, repeated.manifest_sha256)
  assert_true(first.manifest_sha256 != hopper.manifest_sha256)
  assert_true(hopper.manifest_sha256 != blackwell.manifest_sha256)
  assert_eq(first.simple.source_digest(), repeated.simple.source_digest())
  assert_eq(first.gated.source_digest(), repeated.gated.source_digest())
  assert_eq(first.simple.workspace_bytes(), 4L)
  assert_eq(first.gated.workspace_bytes(), 8L)
  let manifest = try! @utf8.decode(first.manifest)
  assert_true(manifest.contains("target=sm_89\n"))
  assert_true(manifest.contains("compiler_version=13.1.115\n"))
  assert_true(manifest.contains("generated_lowerings=1\n"))
  assert_true(manifest.contains("runtime_authority=0\n"))
  assert_true(manifest.contains("numerical_execution=0\nreadiness=0\n"))
  let blackwell_manifest = try! @utf8.decode(blackwell.manifest)
  assert_true(blackwell_manifest.contains("target=sm_120\n"))
  assert_true(blackwell_manifest.contains("runtime_authority=0\n"))
  assert_true(
    blackwell_manifest.contains("numerical_execution=0\nreadiness=0\n"),
  )
}

///|
test "compile candidates reject unsupported target and malformed digest" {
  let digest = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
  try prepare_compile_candidates("sm_100", digest) catch {
    InvalidArguments => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("unsupported compile candidate was admitted")
  }
  try prepare_compile_candidates("sm_89", "not-a-digest") catch {
    InvalidArguments => ()
    _ => fail("unexpected digest rejection")
  } noraise {
    _ => fail("malformed compiler digest was admitted")
  }
}

///|
test "simple and gated identities cannot alias" {
  let set = try! prepare_compile_candidates(
    "sm_90", "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  assert_true(set.simple.source_digest() != set.gated.source_digest())
  assert_true(set.simple.recipe_digest() != set.gated.recipe_digest())
  assert_true(set.simple.function_symbol() != set.gated.function_symbol())
  assert_eq(set.simple.operation_id(), try! @plan.OperationId::new(2))
  assert_eq(set.gated.operation_id(), try! @plan.OperationId::new(8))
}
