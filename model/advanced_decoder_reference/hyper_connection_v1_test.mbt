///|
test "versioned block controls and pre/post reductions follow official shapes" {
  let limits = try! ReferenceLimits::new(
    max_values=4096,
    max_width=64,
    max_iterations=32,
  )
  let streams : Array[Float] = [1.0F, 2.0F, 3.0F, 4.0F]
  let controls = hyper_connection_block_controls_v1(
    streams,
    Array::make(12 * 4, 0.0F),
    Array::make(12, 0.0F),
    [1.0F, 1.0F, 1.0F],
    row_count=1,
    stream_count=2,
    feature_width=2,
    normalization_epsilon=0.000001F,
    control_epsilon=0.000001F,
    sinkhorn_iterations=3,
    limits~,
  )
  assert_eq(controls.pre(), [0.500001F, 0.500001F])
  assert_eq(controls.post(), [1.0F, 1.0F])
  assert_eq(controls.combination().length(), 4)
  for value in controls.combination() {
    assert_true(value > 0.499F && value < 0.501F)
  }
  let pre = hyper_connection_pre_reduce_v1(
    streams,
    controls.pre(),
    row_count=1,
    stream_count=2,
    feature_width=2,
    limits~,
  )
  assert_true(pre[0] > 2.0F && pre[0] < 2.001F)
  assert_true(pre[1] > 3.0F && pre[1] < 3.001F)
  let post = hyper_connection_post_combine_v1(
    [10.0F, 20.0F],
    streams,
    controls.post(),
    controls.combination(),
    row_count=1,
    stream_count=2,
    feature_width=2,
    limits~,
  )
  assert_eq(post.length(), 4)
  assert_true(post[0] > 11.99F && post[0] < 12.01F)
  assert_true(post[1] > 22.99F && post[1] < 23.01F)
}

///|
test "versioned head reduction and hostile tensor shapes fail closed" {
  let limits = try! ReferenceLimits::new(
    max_values=4096,
    max_width=64,
    max_iterations=32,
  )
  let head = hyper_connection_head_reduce_v1(
    [1.0F, 2.0F, 3.0F, 4.0F],
    Array::make(2 * 4, 0.0F),
    [0.0F, 0.0F],
    1.0F,
    row_count=1,
    stream_count=2,
    feature_width=2,
    normalization_epsilon=0.000001F,
    control_epsilon=0.000001F,
    limits~,
  )
  assert_true(head[0] > 2.0F && head[0] < 2.001F)
  try
    hyper_connection_block_controls_v1(
      [1.0F, 2.0F, 3.0F, 4.0F],
      Array::make(47, 0.0F),
      Array::make(12, 0.0F),
      [1.0F, 1.0F, 1.0F],
      row_count=1,
      stream_count=2,
      feature_width=2,
      normalization_epsilon=0.000001F,
      control_epsilon=0.000001F,
      sinkhorn_iterations=3,
      limits~,
    )
  catch {
    InvalidDimension => ()
    _ => fail("wrong block-control shape rejection")
  } noraise {
    _ => fail("truncated control function was admitted")
  }
  try
    hyper_connection_head_reduce_v1(
      [1.0F, 2.0F, 3.0F, 4.0F],
      Array::make(8, 0.0F),
      [0.0F, 0.0F],
      1.0F,
      row_count=1,
      stream_count=2,
      feature_width=2,
      normalization_epsilon=-0.000001F,
      control_epsilon=0.000001F,
      limits~,
    )
  catch {
    NonFiniteParameter => ()
    _ => fail("wrong head-control parameter rejection")
  } noraise {
    _ => fail("invalid head normalization epsilon was admitted")
  }
}
