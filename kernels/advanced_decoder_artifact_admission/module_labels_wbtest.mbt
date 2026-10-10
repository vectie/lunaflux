///|
test "advanced decoder module labels do not trigger payload checksum scans" {
  let label = @catalog.AotArtifactDigest::from_sha256("f".repeat(64))
  let limits = AdvancedDecoderArtifactAdmissionLimits::new(
    maximum_requirements=1,
    maximum_modules=1,
    maximum_entry_points=1,
    maximum_module_bytes=32L,
    maximum_total_module_bytes=32L,
    maximum_symbol_bytes=64,
    maximum_operands_per_launch=8,
    maximum_canonical_bytes=4096,
  )
  let modules = [
    AdvancedDecoderModuleInput::new(digest=label, module_bytes=b"opaque module"),
  ]
  assert_eq(validate_modules(modules, [label], limits), 13L)
  try
    validate_modules(
      [AdvancedDecoderModuleInput::new(digest=label, module_bytes=b"")],
      [label],
      limits,
    )
  catch {
    EmptyModule(actual) => assert_eq(actual, label)
    _ => fail("unexpected empty-module error")
  } noraise {
    _ => fail("empty module was accepted")
  }
  try
    validate_modules(
      [
        AdvancedDecoderModuleInput::new(
          digest=label,
          module_bytes=Bytes::make(33, 0),
        ),
      ],
      [label],
      limits,
    )
  catch {
    ModuleTooLarge(actual) => assert_eq(actual, label)
    _ => fail("unexpected module size error")
  } noraise {
    _ => fail("oversized module was accepted")
  }
}
