///|
test "residual add lowering is deterministic and closes the pointer ABI" {
  let first = luna_cuda_test_lowering()
  let second = luna_cuda_test_lowering()
  assert_eq(first, second)
  assert_eq(first.width(), 8L)
  assert_eq(first.element_count(), 48L)
  assert_eq(first.operand_byte_count(), 96L)
  assert_eq(first.launch_dimensions().grid_x(), 1)
  assert_eq(first.launch_dimensions().block_x(), 256)
  assert_eq(first.function_symbol(), "lunaflux_luna_residual_add_v1_ep_7")
  let source = try! @utf8.decode(first.source_bytes())
  assert_true(source.contains("index < 48ULL"))
  assert_true(
    source.contains(
      "(const __nv_bfloat16 *left, const __nv_bfloat16 *right, __nv_bfloat16 *output)",
    ),
  )
  assert_false(source.contains("char *"))
  let recipe = try! @utf8.decode(first.recipe_bytes())
  assert_true(recipe.contains("schema=lunaflux-luna-cuda-aot-recipe-v1\n"))
  assert_true(recipe.contains("target=sm_90\n"))
  assert_true(recipe.contains("output=cubin\n"))
  assert_true(recipe.contains("grid=1,1,1\n"))
  assert_true(recipe.contains("block=256,1,1\n"))
  assert_true(recipe.contains("abi=pointer,pointer,pointer\n"))
  assert_false(recipe.contains("--use_fast_math"))
}

///|
test "physical residual fixture is exact typed lowering output" {
  let program = luna_cuda_test_program()
  let lowering = @luna_cuda_aot.lower_luna_residual_add_cuda_aot(
    program,
    luna_cuda_test_planning(program),
    luna_cuda_test_target(),
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=2,
      sequence_tokens=1,
    ),
    @catalog.AotKernelEntryPointId::from_stable_int(7),
    luna_cuda_test_compiler(),
    luna_cuda_test_limits(),
  )
  assert_eq(lowering.element_count(), 16L)
  assert_eq(
    lowering.source_digest().as_hex(),
    "fbac275212bf95fcd71aa3dcf7cd8adc7a07513b47f0b8251945805e68bfe13f",
  )
}

///|
test "deterministic compiled output becomes content-addressed artifact input" {
  let lowering = luna_cuda_test_lowering()
  let cubin = b"ELF-CUBIN-fixture-v1"
  let artifact = @luna_cuda_aot.admit_luna_cuda_aot_output(
    lowering,
    cubin,
    cubin,
    luna_cuda_test_receipt(lowering, cubin),
    luna_cuda_test_limits(),
  )
  assert_eq(artifact.module_bytes(), cubin)
  assert_eq(
    artifact.module_relative_path(),
    "sha256/" + artifact.module_digest().sha256() + ".cubin",
  )
  let evidence = try! @utf8.decode(artifact.evidence_bytes())
  assert_eq(
    artifact.driver_identity_sha256(),
    "3333333333333333333333333333333333333333333333333333333333333333",
  )
  assert_true(evidence.contains("driver_identity_sha256=3333333333333333"))
  assert_true(evidence.contains("deterministic=1\n"))
  assert_true(
    evidence.contains("artifact_sha256=" + artifact.module_digest().sha256()),
  )
}
