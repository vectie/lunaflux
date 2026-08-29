///|
test "admitted FP8 launch ABI binds plan manifest kernel scale and workspace" {
  let fixture = fp8_abi_fixture()
  let abi = admit_fixture(fixture)
  let manifest = fixture.kernels.manifest()
  assert_eq(abi.version(), @fp8_launch_abi.Fp8LaunchAbiVersion::v1())
  assert_eq(abi.scope(), StatelessFullPrefillV1)
  assert_eq(abi.plan_identity(), fixture.plan.identity())
  assert_eq(
    abi.numeric_schema_digest(),
    fixture.plan.numeric_binding().digest(),
  )
  assert_eq(abi.manifest_digest(), manifest.digest())
  assert_eq(abi.target(), manifest.target())
  assert_eq(abi.profile().as_int(), 1)
  assert_eq(abi.operation().as_int(), 1)
  assert_eq(abi.execution_digest(), manifest.entries()[0].execution_digest())
  assert_eq(abi.module_digest(), fixture_module_digest())
  assert_eq(abi.entry_point(), fixture_entry(2, 11))
  assert_eq(abi.exact_shape(), fixture.shape)
  assert_eq(abi.scale_kind(), FiniteE4M3DynamicF32V1)
  assert_eq(
    abi.scale_granularity(),
    @numeric_contract.ScaleGranularity::per_tensor(),
  )
  assert_eq(abi.activation_scale_workspace(), fixture.scale)
  assert_eq(abi.kernel_workspace(), Some(fixture.workspace))
  assert_eq(abi.workspace_bytes(), 64L)
  assert_eq(abi.workspace_alignment(), 16L)
  assert_eq(abi.weight_scales().length(), 1)
  assert_eq(abi.weight_scales()[0].tensor().as_int(), 1)
  assert_eq(abi.weight_scales()[0].scale().tensor().as_int(), 2)
  assert_eq(
    abi.weight_scales()[0].scale_storage_digest(),
    manifest.entries()[0].weights()[0].scale_storage_digest(),
  )
  assert_eq(abi.scale_tensor_regions().length(), 1)
  assert_eq(abi.scale_tensor_regions()[0].tensor().tensor().as_int(), 2)
  assert_eq(
    abi.scale_tensor_regions()[0].storage_digest(),
    manifest.entries()[0].weights()[0].scale_storage_digest(),
  )
  assert_eq(abi.scale_tensor_regions()[0].region(), fixture.weight_scale)
  assert_eq(abi.digest().as_sha256().length(), 64)
  assert_eq(admit_fixture(fp8_abi_fixture()).digest(), abi.digest())
  inspect(
    abi.digest().as_sha256(),
    content="d31b4d800434292b118bd575deaedf97bce315c6c04fee48b0a6d35a7feab433",
  )
}

///|
test "region identity and exact offset contribute to ABI identity" {
  let fixture = fp8_abi_fixture()
  let baseline = admit_fixture(fixture)
  let shifted = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=fixture.workspace.storage(),
    storage_bytes=fixture.workspace.storage_bytes(),
    offset=64L,
    byte_length=64L,
    alignment=16L,
  )
  let changed = admit_fixture(fixture, workspace=shifted)
  assert_true(changed.digest() != baseline.digest())
  assert_eq(changed.kernel_workspace(), Some(shifted))
}

///|
test "owner grants one lease and enforces acquire use release order" {
  let fixture = fp8_abi_fixture()
  let abi = admit_fixture(fixture)
  let owner = try! @fp8_launch_abi.Fp8DynamicLaunchOwner::new(
    abi,
    generation=9UL,
  )
  assert_eq(owner.state(), Available)
  let lease = owner.acquire(abi, generation=9UL)
  assert_eq(owner.state(), Acquired)
  try owner.acquire(abi, generation=9UL) catch {
    InvalidFp8LaunchAbi(InvalidTransition) => ()
    _ => fail("unexpected second-lease failure")
  } noraise {
    _ => fail("second live lease was issued")
  }
  lease.begin_use(abi, generation=9UL)
  assert_eq(owner.state(), InUse)
  lease.finish_use()
  assert_eq(owner.state(), Acquired)
  lease.release()
  lease.release()
  assert_eq(owner.state(), Available)
  owner.close()
  owner.close()
  assert_eq(owner.state(), Closed)
}

///|
test "stale lease and foreign ABI binding fail closed" {
  let fixture = fp8_abi_fixture()
  let abi = admit_fixture(fixture)
  let foreign_scale = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
    storage_bytes=16L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  let foreign = admit_fixture(fixture, scale=foreign_scale)
  let owner = try! @fp8_launch_abi.Fp8DynamicLaunchOwner::new(
    abi,
    generation=11UL,
  )
  try owner.acquire(foreign, generation=11UL) catch {
    InvalidFp8LaunchAbi(BindingMismatch) => ()
    _ => fail("unexpected foreign-binding failure")
  } noraise {
    _ => fail("foreign ABI binding acquired authority")
  }
  try owner.acquire(abi, generation=12UL) catch {
    InvalidFp8LaunchAbi(BindingMismatch) => ()
    _ => fail("unexpected generation-binding failure")
  } noraise {
    _ => fail("foreign generation acquired authority")
  }
  let old = owner.acquire(abi, generation=11UL)
  old.release()
  let current = owner.acquire(abi, generation=11UL)
  try old.begin_use(abi, generation=11UL) catch {
    InvalidFp8LaunchAbi(StaleLease) => ()
    _ => fail("unexpected stale-lease failure")
  } noraise {
    _ => fail("stale lease entered use")
  }
  current.release()
}

///|
test "terminal close remains idempotent after release fails in use" {
  let fixture = fp8_abi_fixture()
  let abi = admit_fixture(fixture)
  let owner = try! @fp8_launch_abi.Fp8DynamicLaunchOwner::new(
    abi,
    generation=13UL,
  )
  let lease = owner.acquire(abi, generation=13UL)
  lease.begin_use(abi, generation=13UL)
  try lease.release() catch {
    InvalidFp8LaunchAbi(InvalidTransition) => ()
    _ => fail("unexpected in-use release failure")
  } noraise {
    _ => fail("in-use lease was released normally")
  }
  owner.close()
  owner.close()
  lease.release()
  assert_eq(owner.state(), Closed)
}
