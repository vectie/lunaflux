///|
test "current compiler retains physical fixture sources with current recipes" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "8bf155ac2033793128053d7d28d96054e8b607b8169c5378daf258e297616b4a", "7cb12e0e7b31bc4dce14bf3659019de35ead821ce7cd1faf7c3d1be8ac6b8e29",
    "80bd64c0c4607f993f04f5ec0fd489959bd96ba430b00a09f952a3324b5aa6d5", "03f70f7c91a9416be35d64edeeeee9ce805446523210ae74a5f15627bd5669d6",
  ]
  let recipe_sha256 = [
    // Compiler canonical forms have evolved since these physical source
    // fixtures were recorded. Source hashes stay pinned; these recipe hashes
    // describe current compilation, not a new physical qualification.
    "f499ebe600cd830d364c7439896bca84d676a53314a3abc3a706039208b00430", "615a6a1a7170cb6f23bab198e724db7d64fb0d10337ead7f1a0c58edeb89659f",
    "fc3477d46718c8ad9c45fb91a70c618d7e3715dc242061533bd57d358e5c1fca", "e7d31acd21d0b63fc02c2f5f80f0dbc0fb3d1e6418733150c97c3966689cdbfe",
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
