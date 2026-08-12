///|
test "embedding and dense projection use declared row-major semantics" {
  let limits = reference_limits()
  let table = @reference.Tensor::new(
    rows=3,
    columns=2,
    values=[1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F],
    limits~,
  )
  let embedded = @reference.embedding([2, 0], table, limits~)
  assert_activation_close(embedded, 2, 2, [5.0F, 6.0F, 1.0F, 2.0F])
  let weight = @reference.Tensor::new(
    rows=3,
    columns=2,
    values=[1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F],
    limits~,
  )
  let input = @reference.Activation::new(
    rows=1,
    columns=2,
    values=[1.0F, 2.0F],
    limits~,
  )
  assert_activation_close(
    @reference.dense_projection(input, weight, limits~),
    1,
    3,
    [1.0F, 2.0F, 3.0F],
  )
  assert_activation_close(
    @reference.language_model_head(input, weight, limits~),
    1,
    3,
    [1.0F, 2.0F, 3.0F],
  )
}

///|
test "RMS normalization accumulates in Float and applies the scale" {
  let limits = reference_limits()
  let input = @reference.Activation::new(
    rows=1,
    columns=2,
    values=[3.0F, 4.0F],
    limits~,
  )
  let weight = @reference.Tensor::new(
    rows=1,
    columns=2,
    values=[1.0F, 1.0F],
    limits~,
  )
  assert_activation_close(
    @reference.rms_norm(input, weight, epsilon=1.0e-5F, limits~),
    1,
    2,
    [0.8485277F, 1.1313703F],
    tolerance=2.0e-5F,
  )
}

///|
test "QKV projection concatenates Q then K then V" {
  let limits = reference_limits()
  let input = @reference.Activation::new(
    rows=1,
    columns=2,
    values=[1.0F, 2.0F],
    limits~,
  )
  let query = @reference.Tensor::new(
    rows=2,
    columns=2,
    values=[1.0F, 0.0F, 0.0F, 1.0F],
    limits~,
  )
  let key = @reference.Tensor::new(
    rows=1,
    columns=2,
    values=[1.0F, 1.0F],
    limits~,
  )
  let value = @reference.Tensor::new(
    rows=1,
    columns=2,
    values=[2.0F, 0.0F],
    limits~,
  )
  assert_activation_close(
    @reference.qkv_projection(input, query, key, value, limits~),
    1,
    4,
    [1.0F, 2.0F, 3.0F, 2.0F],
  )
}

///|
test "rotary embedding rotates Q and K while leaving V unchanged" {
  let limits = reference_limits()
  let qkv = @reference.Activation::new(
    rows=1,
    columns=6,
    values=[1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 3.0F],
    limits~,
  )
  assert_activation_close(
    @reference.rotary_embedding(
      qkv,
      query_heads=1,
      kv_heads=1,
      head_dimension=2,
      position_offset=1,
      rope_theta=10000.0F,
      limits~,
    ),
    1,
    6,
    [0.5403023F, 0.8414710F, -0.8414710F, 0.5403023F, 2.0F, 3.0F],
    tolerance=2.0e-5F,
  )
}

///|
test "Llama rotary embedding uses split-half rather than adjacent pairs" {
  let limits = reference_limits()
  let qkv = @reference.Activation::new(
    rows=1,
    columns=12,
    values=[
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F,
    ],
    limits~,
  )
  assert_activation_close(
    @reference.rotary_embedding(
      qkv,
      query_heads=1,
      kv_heads=1,
      head_dimension=4,
      position_offset=1,
      rope_theta=10000.0F,
      limits~,
    ),
    1,
    12,
    [
      -1.9841106F,
      1.9599007F,
      2.4623778F,
      4.0197997F,
      -3.1887856F,
      5.919701F,
      7.9894705F,
      8.059599F,
      9.0F,
      10.0F,
      11.0F,
      12.0F,
    ],
    tolerance=3.0e-5F,
  )
}

///|
test "causal GQA maps two query heads to one KV head" {
  let limits = reference_limits()
  let qkv = @reference.Activation::new(
    rows=2,
    columns=8,
    values=[
      0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
      0.0F, 4.0F, 8.0F,
    ],
    limits~,
  )
  assert_activation_close(
    @reference.causal_gqa_attention(
      qkv,
      query_heads=2,
      kv_heads=1,
      head_dimension=2,
      limits~,
    ),
    2,
    4,
    [2.0F, 4.0F, 2.0F, 4.0F, 3.0F, 6.0F, 3.0F, 6.0F],
  )
}

///|
test "residual and SiLU-gated MLP have explicit dense-Llama semantics" {
  let limits = reference_limits()
  let one = @reference.Activation::new(
    rows=1,
    columns=1,
    values=[1.0F],
    limits~,
  )
  let two = @reference.Activation::new(
    rows=1,
    columns=1,
    values=[2.0F],
    limits~,
  )
  assert_activation_close(@reference.residual_add(one, two, limits~), 1, 1, [
    3.0F,
  ])
  let gate = @reference.Tensor::new(rows=1, columns=1, values=[1.0F], limits~)
  let up = @reference.Tensor::new(rows=1, columns=1, values=[2.0F], limits~)
  let down = @reference.Tensor::new(rows=1, columns=1, values=[3.0F], limits~)
  assert_activation_close(
    @reference.silu_gated_mlp(one, gate, up, down, limits~),
    1,
    1,
    [4.3863516F],
    tolerance=2.0e-5F,
  )
}
