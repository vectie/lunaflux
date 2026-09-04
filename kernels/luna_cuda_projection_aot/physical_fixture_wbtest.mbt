///|
test "physical sm120 projection fixtures bind exact lowering digests" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "80bd64c0c4607f993f04f5ec0fd489959bd96ba430b00a09f952a3324b5aa6d5", "03f70f7c91a9416be35d64edeeeee9ce805446523210ae74a5f15627bd5669d6",
  ]
  let recipe_sha256 = [
    "968f3a45428665dfaceec3a608902959c6ccac34713adb8d368c616665594e8b", "9c8ea54f7805162450248358301e5d365032b31572169d040bc91aead30e99c5",
    "82f33a1ff5eeca671081e7955378f3c8a28299dd0a522e06b35a64ade0c53251", "a2cc690255d648b126aa04ad4c323682067336d93bbccb404adc28b7bcbb11a9",
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
