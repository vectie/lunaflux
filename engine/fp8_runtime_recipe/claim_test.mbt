///|
fn claim_region(
  storage : String,
  offset? : Int64 = 0L,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion {
  try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(storage),
    storage_bytes=64L,
    offset~,
    byte_length=4L,
    alignment=4L,
  )
}

///|
test "weight operand claim is bounded before copying caller regions" {
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation = try! @plan.OperationId::new(0)
  let region = claim_region(
    "1111111111111111111111111111111111111111111111111111111111111111",
  )
  for regions in [[], [region, region], [region, region, region, region]] {
    try
      @fp8_runtime_recipe.Fp8LaunchWeightRegionClaim::new(
        profile~,
        operation~,
        regions~,
      )
    catch {
      InvalidFp8RuntimeRecipe(WeightRegion) => ()
      _ => fail("unexpected weight-region claim failure")
    } noraise {
      _ => fail("unbounded weight-region claim was retained")
    }
  }
  ignore(
    try! @fp8_runtime_recipe.Fp8LaunchWeightRegionClaim::new(
      profile~,
      operation~,
      regions=[region],
    ),
  )
  ignore(
    try! @fp8_runtime_recipe.Fp8LaunchWeightRegionClaim::new(
      profile~,
      operation~,
      regions=[region, region, region],
    ),
  )
}

///|
test "v2 weight operand claim is bounded before copying caller regions" {
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation = try! @plan.OperationId::new(0)
  let region = claim_region(
    "2222222222222222222222222222222222222222222222222222222222222222",
  )
  for regions in [[], [region, region], [region, region, region, region]] {
    try
      @fp8_runtime_recipe.Fp8LaunchWeightRegionClaimV2::new(
        profile~,
        operation~,
        regions~,
      )
    catch {
      InvalidFp8RuntimeRecipeV2(WeightRegion) => ()
      _ => fail("unexpected v2 weight-region claim failure")
    } noraise {
      _ => fail("unbounded v2 weight-region claim was retained")
    }
  }
}
