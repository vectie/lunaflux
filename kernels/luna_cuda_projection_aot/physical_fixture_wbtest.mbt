///|
test "physical sm120 projection fixtures bind exact lowering digests" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "5c22598f9d24ee9a6dbecf70d858814d040a579a565ea414b691099d9815aa3f", "54b51660051c22e98fa8be726a3c3408b40635d458e4ab087a43f6665a41b44f",
  ]
  let recipe_sha256 = [
    "07b0ea49b1d230dfe5de1c1a9542976a5ff773afe276cb7bece3b7f58a84cdfd", "fa15150d817abdcd77881bfc2d84003753320b8f5287c8bbd140b27a2610f323",
    "14a076c53c1570310c0754257752770e5273b501c65bb552217a8904dcbccfbe", "71296d1314c61d4c94eae4e0efbe327f1cb424b2bcedb9bd4ae8b402d7a82ede",
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
