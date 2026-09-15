///|
test "unified scalar source and recipe snapshots remain deterministic" {
  let cases = [("qkv", 1), ("dense", 4), ("mlp", 5), ("lm_head", 6)]
  let source_sha256 = [
    "55e13aadc1ec8da702754fedee3fcc1400fe4e17e2fbb8b81893ad79db2e404d", "cd7829586c606ededfad491f9c232a3e651f64e15c35b1c0612229e6859eacc9",
    "d9f0331fcd68b053fb7bb262a1a7d2fc85f4b28ff9dd78c9204a2a83a7c12321", "869c6fb18964c5b26bd359306b100dbbf4dbaa7df7e473e04176d33c430c624b",
  ]
  let recipe_sha256 = [
    // These are compiler snapshots, not transitive physical qualifications.
    "254e5699e7781c467c4711d16cfa981aaa5cb91761a92a8a37112cb2c933384a", "44da19a6f299b0d7bd4cc0dd384ca3fe4cef6066e355587835a4fa6b1cbc8de4",
    "5bc9f61ab2f1f2295e336727ecb020bc37864083260aa574d247025ee4516fb4", "9b0a0f517675effc47dfcb14b285bf1912ea91829d50f64a0aa1ef8023e434ba",
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
    if @env.get_env_var("LUNA_TEST_EXPORT_SCALAR") == Some("1") {
      println(
        "LUNA_DIGEST \{name} \{lowering.source_digest().as_hex()} \{lowering.recipe_digest().as_hex()}",
      )
      println("LUNA_SOURCE_BEGIN scalar-\{name}")
      println(@utf8.decode(lowering.source_bytes()))
      println("LUNA_SOURCE_END")
    }
    assert_eq(lowering.source_digest().as_hex(), source_sha256[index])
    assert_eq(lowering.recipe_digest().as_hex(), recipe_sha256[index])
    let _ = name
    let recipe = @utf8.decode(lowering.recipe_bytes())
    assert_true(recipe.contains("target=sm_120\n"))
    assert_true(recipe.contains("source_sha256=" + source_sha256[index] + "\n"))
  }
}
