///|
fn source_input(
  digest : @catalog.AotArtifactDigest,
  path : String,
) -> @artifact_file.KernelModuleFileInput {
  @artifact_file.KernelModuleFileInput::new(digest~, path_identifier=path)
}

///|
test "direct module-file source admission is bounded and inert" {
  let limits = fixture_file_limits()
  let digest = fixture_digest()
  let source = @artifact_file.KernelArtifactManifest::admit_files(
    [source_input(digest, "sm90/kernel.cubin")],
    [],
    limits,
  )
  assert_eq(source.module_count(), 1)
  assert_eq(source.entry_point_count(), 0)
}

///|
test "direct module-file source defensively owns collection membership" {
  let digest = fixture_digest()
  let modules = [source_input(digest, "kernel.cubin")]
  let entries = [
    @artifact.KernelEntryPointInput::new(
      entry_point=fixture_entry(digest, 1),
      function_symbol="entry",
    ),
  ]
  let source = @artifact_file.KernelArtifactManifest::admit_files(
    modules,
    entries,
    fixture_file_limits(),
  )
  modules.clear()
  entries.clear()
  assert_eq(source.module_count(), 1)
  assert_eq(source.entry_point_count(), 1)
}

///|
test "direct module-file source rejects unsafe and duplicate locators" {
  let limits = fixture_file_limits()
  let digest = fixture_digest()
  try
    @artifact_file.KernelArtifactManifest::admit_files(
      [source_input(digest, "../kernel.cubin")],
      [],
      limits,
    )
  catch {
    InvalidPath(file=ModuleFile(0), issue=TraversalSegment) => ()
    _ => fail("unexpected unsafe-locator error")
  } noraise {
    _ => fail("expected unsafe locator rejection")
  }
  try
    @artifact_file.KernelArtifactManifest::admit_files(
      [
        source_input(digest, "kernel.cubin"),
        source_input(digest, "kernel.cubin"),
      ],
      [],
      limits,
    )
  catch {
    InvalidPath(file=ModuleFile(1), issue=DuplicatePath) => ()
    _ => fail("unexpected duplicate-locator error")
  } noraise {
    _ => fail("expected duplicate locator rejection")
  }
}

///|
test "direct module-file source enforces module collection bounds" {
  let bounded = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=4096L,
    max_json_depth=8,
    max_path_bytes=128,
    artifact=@artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=8,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
      max_function_symbol_bytes=128,
    ),
  )
  let digest = fixture_digest()
  try
    @artifact_file.KernelArtifactManifest::admit_files(
      [source_input(digest, "one.cubin"), source_input(digest, "two.cubin")],
      [],
      bounded,
    )
  catch {
    SourceCountExceeded(source=SourceModules, maximum=1) => ()
    _ => fail("unexpected source-count error")
  } noraise {
    _ => fail("expected source module bound rejection")
  }
}

///|
test "direct module-file source enforces entry-point collection bounds" {
  let bounded = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=4096L,
    max_json_depth=8,
    max_path_bytes=128,
    artifact=@artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=1,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
      max_function_symbol_bytes=128,
    ),
  )
  let digest = fixture_digest()
  let inputs = [
    @artifact.KernelEntryPointInput::new(
      entry_point=fixture_entry(digest, 1),
      function_symbol="one",
    ),
    @artifact.KernelEntryPointInput::new(
      entry_point=fixture_entry(digest, 2),
      function_symbol="two",
    ),
  ]
  try
    @artifact_file.KernelArtifactManifest::admit_files(
      [source_input(digest, "kernel.cubin")],
      inputs,
      bounded,
    )
  catch {
    SourceCountExceeded(source=SourceEntryPoints, maximum=1) => ()
    _ => fail("unexpected entry-point count error")
  } noraise {
    _ => fail("expected source entry-point bound rejection")
  }
}
