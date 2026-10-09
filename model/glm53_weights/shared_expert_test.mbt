///|
test "installed GLM shared experts keep BF16 and no scale planes" {
  for layer in [3, 44, 45] {
    let weights = @glm53_weights.shared_expert_weights(layer_index=layer)
    assert_eq(weights.length(), 3)
    assert_eq(
      weights[0].payload_name(),
      "model.language_model.layers.\{layer}.mlp.shared_experts.gate_proj.weight",
    )
    assert_eq(weights[2].representation().shape(), (4096, 2048))
    for weight in weights {
      assert_eq(weight.layer_index(), layer)
      assert_eq(weight.representation().format(), Bf16)
      assert_eq(weight.representation().scale_bytes(), 0L)
      assert_eq(weight.representation().total_bytes(), 16777216L)
    }
  }
}
