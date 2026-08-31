///|
test "physical sm120 pointwise fixtures bind exact lowering digests" {
  let profile = pointwise_test_profile()
  let cases = [
    (
      "embedding",
      pointwise_embedding_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ExternalInput(TokenIds), 16L, alignment=4L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(0)), 256L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(0)), 64L),
      ],
    ),
    (
      "rms_norm",
      pointwise_rms_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(0)), 64L),
        pointwise_operand(WeightInput(@plan.TensorRef::new(1)), 16L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(1)), 64L),
      ],
    ),
    (
      "residual",
      pointwise_residual_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(0)), 64L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(1)), 64L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(2)), 64L),
      ],
    ),
    (
      "rope",
      pointwise_rotary_operation(),
      [
        pointwise_operand(RuntimeInput(StepCounts), 20L, alignment=4L),
        pointwise_operand(RuntimeInput(QueryPositions), 16L, alignment=4L),
        pointwise_operand(ActivationInput(@plan.ActivationRef::new(2)), 128L),
        pointwise_operand(ActivationOutput(@plan.ActivationRef::new(3)), 128L),
      ],
    ),
  ]
  let source_sha256 = [
    "129a53ac394ca6a444c5eac67cfe9c9ebd2eaac44b624525a9ac0843ea0847fe", "805fa6114824c130fba685ae2b767f44a2c11b49740549c99ba7b5dcc358b9aa",
    "1c1ee8ff5fafe40b23595ff27c8d790bd2db70a349bcb3c2bb861e357468827a", "66277b5fad86db509aac8242df094271f72f4b67c6e2da6c1bbd6d02c007001c",
  ]
  let recipe_sha256 = [
    "df9eb0c186351ed13b404f1cba40bd7da60adbe017a58b3e866e02bcdc92ed63", "4bc8f06a11e5d5119e08929449f46c4d5ab5bb063f6e9878740297029d3037c6",
    "8da723aa9e542b7a1f08b6c916ab51024a1c663f67caf46ccad65caa7178eca6", "592a735b2cb4b558d1b936fd83ba067e6c973d35e23b0401752adf2bc7be7372",
  ]
  for index, case in cases {
    let (name, operation, operands) = case
    let lowering = lower_pointwise_cuda_aot_candidate(
      operation,
      profile,
      pointwise_test_entry(index + 1),
      @catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=0,
        supports_bf16=true,
        supports_cublas_lt=true,
      ),
      pointwise_test_compiler(),
      operands,
      pointwise_test_limits(),
    )
    assert_eq(lowering.source_digest().as_hex(), source_sha256[index])
    assert_eq(lowering.recipe_digest().as_hex(), recipe_sha256[index])
    let _ = name
    let recipe = @utf8.decode(lowering.recipe_bytes())
    assert_true(recipe.contains("target=sm_120\n"))
    assert_true(recipe.contains("source_sha256=" + source_sha256[index] + "\n"))
  }
}
