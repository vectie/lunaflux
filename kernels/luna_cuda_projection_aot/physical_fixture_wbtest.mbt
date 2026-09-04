///|
test "physical sm120 projection fixtures bind exact lowering digests" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "80bd64c0c4607f993f04f5ec0fd489959bd96ba430b00a09f952a3324b5aa6d5", "03f70f7c91a9416be35d64edeeeee9ce805446523210ae74a5f15627bd5669d6",
  ]
  let recipe_sha256 = [
    "0489f1771b25e9635049871e1fe3cf4dfa0904bf86663b385c696d1ac36fb971", "2e15613d273c678599c3cb3181ac12b5abed58b9bc2451f934a6a90d5e0525cc",
    "f0e3b894ed2d555cf55cc41015ffe48f09d68b211cc82d81f8a1345875322b1b", "2033a69c223227cdb3c9f413c06a83ab728bcd057b9eb5883f51a11485402f56",
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
