///|
test "NVFP4 expert partitions retain compact official projection shapes and names" {
  let weights = @glm53_weights.nvfp4_expert_weights(layer_index=10, expert_indices=[
    287, 0,
  ])
  assert_eq(weights.length(), 6)
  assert_eq(
    weights[0].payload_name(),
    "model.language_model.layers.10.mlp.experts.287.gate_proj.weight",
  )
  assert_eq(
    weights[0].scales_name(),
    "model.language_model.layers.10.mlp.experts.287.gate_proj.weight_scale",
  )
  assert_eq(
    weights[0].global_scale_name(),
    "model.language_model.layers.10.mlp.experts.287.gate_proj.weight_scale_2",
  )
  assert_eq(weights[0].representation().shape(), (2048, 4096))
  assert_eq(weights[2].representation().shape(), (4096, 2048))
  assert_eq(weights[0].representation().scale_shape(), (2048, 256))
  assert_eq(weights[2].representation().scale_shape(), (4096, 128))
  for weight in weights {
    assert_eq(weight.layer_index(), 10)
    assert_eq(weight.representation().payload_bytes(), 4194304L)
    assert_eq(weight.representation().scale_bytes(), 524288L)
    assert_eq(weight.representation().total_bytes(), 4718596L)
  }
  assert_eq(weights[3].expert_index(), 0)
  let mtp = @glm53_weights.nvfp4_expert_weights(layer_index=45, expert_indices=[
    0,
  ])
  assert_eq(
    mtp[0].payload_name(),
    "model.language_model.layers.45.mlp.experts.0.gate_proj.weight",
  )
}

///|
test "NVFP4 partitions exclude dense layers and invalid or repeated experts" {
  for layer in [0, 2, 46] {
    try
      @glm53_weights.nvfp4_expert_weights(layer_index=layer, expert_indices=[0])
    catch {
      InvalidMoeLayer(actual) => assert_eq(actual, layer)
      _ => fail("wrong layer error")
    } noraise {
      _ => fail("non-MoE layer accepted")
    }
  }
  for expert in [-1, 288] {
    try
      @glm53_weights.nvfp4_expert_weights(layer_index=3, expert_indices=[expert])
    catch {
      InvalidExpert(actual) => assert_eq(actual, expert)
      _ => fail("wrong expert error")
    } noraise {
      _ => fail("invalid expert accepted")
    }
  }
  try
    @glm53_weights.nvfp4_expert_weights(layer_index=3, expert_indices=[1, 1])
  catch {
    DuplicateExpert(1) => ()
    _ => fail("wrong duplicate error")
  } noraise {
    _ => fail("same expert uploaded twice")
  }
}
