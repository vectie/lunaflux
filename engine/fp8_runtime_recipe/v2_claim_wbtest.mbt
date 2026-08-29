///|
test "v2 weight operand claim detaches the caller array" {
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation = try! @plan.OperationId::new(0)
  let region = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
    storage_bytes=64L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  let source = [region]
  let retained = try! Fp8LaunchWeightRegionClaimV2::new(
    profile~,
    operation~,
    regions=source,
  )
  source.push(region)
  assert_eq(retained.regions.length(), 1)
}
