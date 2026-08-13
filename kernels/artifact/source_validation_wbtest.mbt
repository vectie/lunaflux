///|
fn source_entry(
  family : Int,
  entry : Int,
  symbol : String,
) -> KernelEntryPointInput raise {
  KernelEntryPointInput::new(
    entry_point=@catalog.AotKernelEntryPoint::new(
      family=@catalog.AotKernelFamily::new(
        module_digest=@catalog.AotArtifactDigest::from_sha256(
          "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
        ),
        family_id=@catalog.AotKernelFamilyId::from_stable_int(family),
      ),
      entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(entry),
    ),
    function_symbol=symbol,
  )
}

///|
test "source symbol validation is bounded and rejects syntax and count" {
  let limits = KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=2,
    max_module_bytes=16L,
    max_total_module_bytes=16L,
    max_function_symbol_bytes=16,
  )
  validate_entry_point_symbols([source_entry(1, 1, "kernel_1")], limits)
  try validate_entry_point_symbols([source_entry(1, 1, "1bad")], limits) catch {
    InvalidFunctionSymbol(rule=InvalidInitial, ..) => ()
    _ => fail("unexpected invalid-symbol failure")
  } noraise {
    _ => fail("invalid symbol was accepted")
  }
  try
    validate_entry_point_symbols(
      [
        source_entry(1, 1, "a"),
        source_entry(1, 2, "b"),
        source_entry(1, 3, "c"),
      ],
      limits,
    )
  catch {
    EntryPointCountExceeded(maximum=2) => ()
    _ => fail("unexpected count failure")
  } noraise {
    _ => fail("entry count overflow was accepted")
  }
}
