///|
test "physical sm120 projection fixtures bind exact lowering digests" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "80bd64c0c4607f993f04f5ec0fd489959bd96ba430b00a09f952a3324b5aa6d5", "54b51660051c22e98fa8be726a3c3408b40635d458e4ab087a43f6665a41b44f",
  ]
  let recipe_sha256 = [
    "6cd505e5f85e7120e81417a991dd35d0a9b72c53e99e71719f041e6c066582b5", "703b3280e6b7e69894b5e56273a42c7430431778cbfcb97684d6497fca2e6c1f",
    "3b8b9c4785411c4f8cad15eb764c7ec73a8969b1ec06d8212898b16222b47149", "c07aad4ba17c68ed1f749411426b79aa58ce788449a30aecf0aec4cf97672c12",
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
