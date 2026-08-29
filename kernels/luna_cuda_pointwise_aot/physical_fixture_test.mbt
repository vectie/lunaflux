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
    "c221803d92108a138339eedf7b79199a0112640fd612b72570f60b0c1bf4020f", "f5ad19205bdb9f50437ba84149fe06180b42b747b34cc8a7a6b2b2ca77769527",
    "9878f2edd1212440028b2114b5a0778c4e12271d1043f9e9432f50e205bd291f", "73606c6a56c721a91a346fd686438bcfc92e4498de432871c27245578c5a2c20",
  ]
  let recipe_sha256 = [
    "830fd419618b16d1bfc82f7002483878649eb4e012a80641ac9dfb44eb01fe5b", "6a8a821a206dc00c96af6fe8466e4af210df0129cde000e7f3b2f8ad5ca0d600",
    "a9451b953961f27dceb892febb902aeccc79b85af8273b3c213b9d6553bbd323", "87a3e906790195e032b8b8cbef796baa2fb773177d8d96919dd33fd857778cb8",
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
