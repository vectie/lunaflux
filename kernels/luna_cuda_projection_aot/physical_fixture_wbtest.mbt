///|
test "physical sm120 projection fixtures bind exact lowering digests" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "80bd64c0c4607f993f04f5ec0fd489959bd96ba430b00a09f952a3324b5aa6d5", "54b51660051c22e98fa8be726a3c3408b40635d458e4ab087a43f6665a41b44f",
  ]
  let recipe_sha256 = [
    "93884eb8f37e63b93e95eabf8928232ffdc2d26884b3b56edf1960003a3f3156", "f14ef6f9df776114aba83bbe9dd253e5c87f5835929aa23ce56eb81af7ee169f",
    "e7149f999a770cce58a15d2da80b0809fbfa28e5a5d4f6c73e39d1f2c836e05b", "386d8115597e0b80a202b45cdaf908e19f6a35830fb40f54153d0decda24f460",
  ]
  for index, case in cases {
    let (name, operation) = case
    let plan = projection_fixture_plan()
    let contracts = projection_fixture_contracts()
    let contract = contracts.contracts()[operation]
    let lowering = lower_projection_cuda_aot_candidate(
      plan.operations()[operation],
      contract.profile(),
      contract.entry_point().entry_point_id(),
      @catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=0,
        supports_bf16=true,
        supports_cublas_lt=true,
      ),
      projection_fixture_compiler(),
      contract.operands(),
      projection_fixture_limits(),
    )
    assert_eq(lowering.source_digest().as_hex(), source_sha256[index])
    assert_eq(lowering.recipe_digest().as_hex(), recipe_sha256[index])
    let _ = name
    let recipe = @utf8.decode(lowering.recipe_bytes())
    assert_true(recipe.contains("target=sm_120\n"))
    assert_true(recipe.contains("source_sha256=" + source_sha256[index] + "\n"))
  }
}
