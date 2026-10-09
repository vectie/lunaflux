///|
test "embedding emits batch aware channel-major gather and exact suffix" {
  let plan = @video_latent_embedding_cuda_source.render(
    batch=2,
    positions=6,
    input_channels=3,
    hidden_channels=5,
    register_rows=2,
    symbol="embed",
  )
  assert_eq(plan.input_elements, 36L)
  assert_eq(plan.output_elements, 90L)
  let source = @utf8.decode(plan.source)
  assert_true(source.contains("(batch_index * 3ULL + k) * 6ULL + row"))
  assert_true(
    source.contains(
      "row < 8ULL ? registers[(row - 6ULL) * 5ULL + channel] : 0.0f",
    ),
  )
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
}

///|
test "embedding rejects invalid geometry and source injection" {
  for positions in [0, -1, 2147483647] {
    try
      @video_latent_embedding_cuda_source.render(
        batch=1,
        positions~,
        input_channels=24,
        hidden_channels=2048,
        register_rows=4,
        symbol="embed",
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid shape")
    }
  }
  try
    @video_latent_embedding_cuda_source.render(
      batch=1,
      positions=1,
      input_channels=24,
      hidden_channels=2048,
      register_rows=4,
      symbol="bad;symbol",
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("invalid symbol")
  }
}
