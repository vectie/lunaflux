///|
fn validation_limits() -> KernelArtifactFileLimits raise {
  KernelArtifactFileLimits::new(
    max_manifest_bytes=1L,
    max_json_depth=1,
    max_path_bytes=128,
    artifact=@artifact.KernelArtifactLimits::new(
      max_modules=4096,
      max_entry_points=1,
      max_module_bytes=1L,
      max_total_module_bytes=4096L,
      max_function_symbol_bytes=1,
    ),
  )
}

///|
test "defensive source validation rejects a late duplicate path" {
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  let digests : Array[@catalog.AotArtifactDigest] = []
  let paths : Array[String] = []
  for index in 0..<4096 {
    digests.push(digest)
    paths.push("module_\{index}.cubin")
  }
  paths[4095] = "module_0.cubin"
  let forged : KernelArtifactManifest = {
    module_digests: ReadOnlyArray::from_array(digests),
    module_path_identifiers: ReadOnlyArray::from_array(paths),
    entry_points: ReadOnlyArray::from_array([]),
  }
  try validate_admitted_source(forged, validation_limits()) catch {
    InvalidPath(file=ModuleFile(4095), issue=DuplicatePath) => ()
    _ => fail("unexpected defensive late-duplicate error")
  } noraise {
    _ => fail("defensive validation accepted a late duplicate")
  }
}
