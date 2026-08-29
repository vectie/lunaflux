///|
test "storage identities and regions reject malformed arithmetic" {
  try
    @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    )
  catch {
    InvalidFp8LaunchAbi(InvalidStorageIdentity) => ()
    _ => fail("unexpected storage identity failure")
  } noraise {
    _ => fail("uppercase storage identity was admitted")
  }
  let storage = try! @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    fixture_digest("storage"),
  )
  try
    @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage~,
      storage_bytes=64L,
      offset=3L,
      byte_length=4L,
      alignment=4L,
    )
  catch {
    InvalidFp8LaunchAbi(RegionAlignment) => ()
    _ => fail("unexpected alignment failure")
  } noraise {
    _ => fail("misaligned region was admitted")
  }
  try
    @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage~,
      storage_bytes=9223372036854775807L,
      offset=9223372036854775804L,
      byte_length=8L,
      alignment=4L,
    )
  catch {
    InvalidFp8LaunchAbi(RegionOverflow) => ()
    _ => fail("unexpected region overflow failure")
  } noraise {
    _ => fail("overflowing region was admitted")
  }
  try
    @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage~,
      storage_bytes=64L,
      offset=32L,
      byte_length=64L,
      alignment=16L,
    )
  catch {
    InvalidFp8LaunchAbi(RegionOutOfBounds) => ()
    _ => fail("unexpected bounds failure")
  } noraise {
    _ => fail("out-of-bounds region was admitted")
  }
}

///|
test "partial scale and kernel-workspace overlap is rejected" {
  let fixture = fp8_abi_fixture()
  let shared = try! @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    "4444444444444444444444444444444444444444444444444444444444444444",
  )
  let scale = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=shared,
    storage_bytes=128L,
    offset=64L,
    byte_length=4L,
    alignment=4L,
  )
  let workspace = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=shared,
    storage_bytes=128L,
    offset=48L,
    byte_length=64L,
    alignment=16L,
  )
  try admit_fixture(fixture, scale~, workspace~) catch {
    InvalidFp8LaunchAbi(RegionAlias) => ()
    _ => fail("unexpected overlapping-region failure")
  } noraise {
    _ => fail("partially overlapping regions were admitted")
  }
}

///|
test "same storage identity cannot claim conflicting capacities" {
  let fixture = fp8_abi_fixture()
  let shared = try! @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    "5555555555555555555555555555555555555555555555555555555555555555",
  )
  let scale = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=shared,
    storage_bytes=128L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  let workspace = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=shared,
    storage_bytes=256L,
    offset=64L,
    byte_length=64L,
    alignment=16L,
  )
  try admit_fixture(fixture, scale~, workspace~) catch {
    InvalidFp8LaunchAbi(RegionStorageMismatch) => ()
    _ => fail("unexpected storage-capacity failure")
  } noraise {
    _ => fail("conflicting storage capacities were admitted")
  }
}

///|
test "scale and kernel workspace sizes are exact" {
  let fixture = fp8_abi_fixture()
  let scale = fixture_region(0L, 8L, 4L)
  try admit_fixture(fixture, scale~) catch {
    InvalidFp8LaunchAbi(WorkspaceContract) => ()
    _ => fail("unexpected scale-size failure")
  } noraise {
    _ => fail("non-scalar activation scale was admitted")
  }
  let workspace = fixture_region(0L, 32L, 16L, workspace=true)
  try admit_fixture(fixture, workspace~) catch {
    InvalidFp8LaunchAbi(WorkspaceContract) => ()
    _ => fail("unexpected workspace-size failure")
  } noraise {
    _ => fail("undersized kernel workspace was admitted")
  }
}

///|
test "scale tensor regions are exact and cannot alias activation scale" {
  let fixture = fp8_abi_fixture()
  try
    @fp8_launch_abi.admit_dynamic_activation_scale_launch(
      fixture.plan,
      fixture.kernels,
      Stateless(fixture.launches),
      operation=try! @plan.OperationId::new(1),
      shape=fixture.shape,
      weight_scale_regions=[],
      activation_scale_workspace=fixture.scale,
      kernel_workspace=Some(fixture.workspace),
    )
  catch {
    InvalidFp8LaunchAbi(BindingMismatch) => ()
    _ => fail("unexpected missing scale-tensor region failure")
  } noraise {
    _ => fail("missing scale-tensor region was admitted")
  }
  let shared = try! @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    "7777777777777777777777777777777777777777777777777777777777777777",
  )
  let weight_scale = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=shared,
    storage_bytes=16L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  let scale = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=shared,
    storage_bytes=16L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  try admit_fixture(fixture, weight_scale~, scale~) catch {
    InvalidFp8LaunchAbi(RegionAlias) => ()
    _ => fail("unexpected scale-tensor alias failure")
  } noraise {
    _ => fail("aliased scale-tensor regions were admitted")
  }
}

///|
test "shape substitution and unsupported paged scope fail closed" {
  let fixture = fp8_abi_fixture()
  let smaller = try! @fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
    profile=@launch_contract.KernelProfileId::from_stable_int(1),
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=1,
      sequence_tokens=1,
    ),
  )
  try admit_fixture(fixture, shape=smaller) catch {
    InvalidFp8LaunchAbi(BindingMismatch) => ()
    _ => fail("unexpected shape substitution failure")
  } noraise {
    _ => fail("foreign exact shape was admitted")
  }
  let paged = try! @fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
    profile=@launch_contract.KernelProfileId::from_stable_int(1),
    rows=[
      @fp8_launch_abi.Fp8PagedRowShape::new(
        query_tokens=1,
        context_tokens=1,
        page_table_entries=1,
      ),
    ],
  )
  try admit_fixture(fixture, shape=paged) catch {
    InvalidFp8LaunchAbi(ManifestScope) | InvalidFp8LaunchAbi(ManifestEntry) =>
      ()
    _ => fail("unexpected paged-scope failure")
  } noraise {
    _ => fail("paged shape was admitted against stateless authority")
  }
}

///|
test "stateless multi-profile selection is exact and rejects wrong dimensions" {
  let fixture = fp8_abi_fixture()
  let profile_two = try! @launch_contract.KernelProfileId::from_stable_int(2)
  let exact_two = try! @fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
    profile=profile_two,
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=1,
      sequence_tokens=4,
    ),
  )
  let admitted = admit_fixture(fixture, shape=exact_two)
  assert_eq(admitted.profile(), profile_two)
  assert_eq(admitted.exact_shape(), exact_two)
  let wrong_dimensions = try! @fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
    profile=profile_two,
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=1,
      sequence_tokens=2,
    ),
  )
  try admit_fixture(fixture, shape=wrong_dimensions) catch {
    InvalidFp8LaunchAbi(BindingMismatch) => ()
    _ => fail("unexpected wrong-profile dimension failure")
  } noraise {
    _ => fail("profile two admitted profile-one dimensions")
  }
}

///|
test "paged exact shape detaches caller row storage" {
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let original = try! @fp8_launch_abi.Fp8PagedRowShape::new(
    query_tokens=1,
    context_tokens=8,
    page_table_entries=1,
  )
  let replacement = try! @fp8_launch_abi.Fp8PagedRowShape::new(
    query_tokens=2,
    context_tokens=8,
    page_table_entries=1,
  )
  let rows = [original]
  let shape = try! @fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
    profile~,
    rows~,
  )
  rows[0] = replacement
  guard shape.paged_rows() is Some(owned) else {
    fail("paged rows were not retained")
  }
  assert_eq(owned[0], original)
  assert_eq(shape.query_tokens(), 1L)
}

///|
test "paged shape totals reject overflow before publication" {
  try
    @fp8_launch_abi.Fp8PagedRowShape::new(
      query_tokens=0,
      context_tokens=1,
      page_table_entries=1,
    )
  catch {
    InvalidFp8LaunchAbi(InvalidShape) => ()
    _ => fail("unexpected row-shape failure")
  } noraise {
    _ => fail("empty paged query row was admitted")
  }
  let huge = try! @fp8_launch_abi.Fp8PagedRowShape::new(
    query_tokens=2147483647,
    context_tokens=2147483647,
    page_table_entries=1,
  )
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  try
    @fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(profile~, rows=[
      huge, huge,
    ])
  catch {
    InvalidFp8LaunchAbi(ShapeOverflow) => ()
    _ => fail("unexpected paged total overflow failure")
  } noraise {
    _ => fail("overflowing paged totals were admitted")
  }
}
