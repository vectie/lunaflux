///|
test "physical sm120 projection fixtures bind exact lowering digests" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "80bd64c0c4607f993f04f5ec0fd489959bd96ba430b00a09f952a3324b5aa6d5", "54b51660051c22e98fa8be726a3c3408b40635d458e4ab087a43f6665a41b44f",
  ]
  let recipe_sha256 = [
    "2c2f9b64afdf852967139ca8911889db8c669b6548793094feafa15b2a138d94", "f04963603be53a17dd28b66bc1925ac14be2ec419b9b144a1b38d151e10e75a4",
    "2c02a9a3c68fb087f6c09ae70437b13b2adc236504c42b68608f51946174bca2", "cf0dd599e07be40a8d8cbb98f4d8cebe8910aeff9f39b17c3ff5f1536564f140",
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
