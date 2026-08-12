///|
test "aliased module ownership remains deferred past semantic rejection" {
  let bytes = b"multi-function-cubin".to_fixedarray()
  let digest = digest_for(bytes)
  let module_input = @artifact.KernelModuleInput::new(
    digest~,
    module_bytes=bytes,
  )
  let duplicate_exports = exports(digest)
  duplicate_exports[1] = duplicate_exports[0]
  try
    @artifact.admit(
      contracts(digest),
      [module_input],
      duplicate_exports,
      limits(),
    )
  catch {
    DuplicateEntryPoint(_) => ()
    _ => fail("expected semantic entry rejection")
  } noraise {
    _ => fail("expected failure")
  }
  bytes[0] = b'X'
  try
    @artifact.admit(
      contracts(digest),
      [module_input],
      exports(digest),
      limits(),
    )
  catch {
    DigestMismatch(found) => assert_eq(found, digest)
    _ => fail("expected aliased bytes to remain authoritative")
  } noraise {
    _ => fail("expected failure")
  }
}
