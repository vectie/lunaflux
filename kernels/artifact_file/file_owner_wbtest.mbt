///|
test "filesystem failures map without paths or native payloads" {
  assert_eq(
    filesystem_error(Busy(Close), ModuleFile(2), CloseFile),
    FileIo(file=ModuleFile(2), operation=CloseFile, issue=Busy),
  )
  assert_eq(
    filesystem_error(ChangedDuringRead, ModuleFile(1), SnapshotFile),
    FileChangedDuringRead(ModuleFile(1)),
  )
}

///|
fn primary_preservation_probe() -> Int raise KernelArtifactFileError {
  let primary = DigestMismatch(ModuleFile(0))
  raise_primary_after_cleanup(primary, () => {
    raise FileIo(file=ModuleFile(0), operation=CloseFile, issue=Failed)
  })
}

///|
test "cleanup failure cannot mask a primary artifact failure" {
  try primary_preservation_probe() catch {
    DigestMismatch(ModuleFile(0)) => ()
    _ => fail("cleanup failure masked the primary failure")
  } noraise {
    _ => fail("expected the primary failure")
  }
}
