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
  let bundle = @artifact.admit(
    contracts(digest),
    [module_input],
    exports(digest),
    limits(),
  )
  let retained = bundle.modules()[0].module_bytes()
  assert_eq(retained[0], b'X')
  assert_eq(bundle.modules()[0].digest(), digest)
  bytes[0] = b'Y'
  assert_eq(retained[0], b'X')
}
