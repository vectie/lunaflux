///|
fn workspace_region(
  storage : String,
  capacity? : Int64 = 64L,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion {
  try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(storage),
    storage_bytes=capacity,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
}

///|
test "sequential launches may reuse one exact workspace region only" {
  let identity = try! @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    "2222222222222222222222222222222222222222222222222222222222222222",
  )
  let shared = workspace_region(identity.as_sha256())
  // Reuse is deliberate: launches are ordered and never concurrent in this
  // recipe. Both observations must still name the same future allocation.
  require_workspace_region(shared, identity, 64L)
  require_workspace_region(shared, identity, 64L)
  try require_workspace_region(shared, identity, 128L) catch {
    InvalidFp8RuntimeRecipe(WorkspaceCapacity) => ()
    _ => fail("unexpected workspace-capacity failure")
  } noraise {
    _ => fail("cross-launch workspace capacity drift was admitted")
  }
  let foreign = workspace_region(
    "3333333333333333333333333333333333333333333333333333333333333333",
  )
  try require_workspace_region(foreign, identity, 64L) catch {
    InvalidFp8RuntimeRecipe(WorkspaceStorage) => ()
    _ => fail("unexpected workspace-identity failure")
  } noraise {
    _ => fail("cross-launch workspace identity substitution was admitted")
  }
}
