///|
test "block control source fixes exact bounded V1 ABI and output order" {
  let rendered = block_control_v1_bf16_f32_source(
    stream_width=4,
    hidden_width=4096,
    maximum_rows=2,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    function_symbol="lunaflux_hc_block_control_v1",
  )
  assert_eq(rendered.operation(), BlockControlV1)
  assert_eq(rendered.numerical_contract(), OrderedF32BlockControlV1)
  assert_eq(rendered.stream_width(), 4)
  assert_eq(rendered.hidden_width(), 4096)
  assert_eq(rendered.maximum_rows(), 2)
  assert_eq(rendered.sinkhorn_iterations(), 20)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const int32_t live_rows = counts[3]"))
  assert_true(source.contains("const __nv_bfloat16 *streams"))
  assert_true(source.contains("const float *function_weights"))
  assert_true(source.contains("float *pre, float *post, float *combination"))
  assert_true(source.contains("float mixes[24]"))
  assert_true(source.contains("iteration < 20"))
  assert_true(
    source.contains("destination * 4ULL + (unsigned long long)source"),
  )
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__fdiv_rn"))
  assert_true(source.contains("expf"))
  assert_true(source.contains("sqrtf"))
}

///|
test "block control source rejects unbounded geometry parameters and symbols" {
  try
    block_control_v1_bf16_f32_source(
      stream_width=17,
      hidden_width=4096,
      maximum_rows=2,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      function_symbol="valid",
    )
  catch {
    InvalidStreamWidth => ()
    _ => fail("wrong stream-width rejection")
  } noraise {
    _ => fail("unbounded stream width was admitted")
  }
  try
    block_control_v1_bf16_f32_source(
      stream_width=4,
      hidden_width=65537,
      maximum_rows=2,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      function_symbol="valid",
    )
  catch {
    InvalidHiddenWidth => ()
    _ => fail("wrong hidden-width rejection")
  } noraise {
    _ => fail("unbounded hidden width was admitted")
  }
  try
    block_control_v1_bf16_f32_source(
      stream_width=4,
      hidden_width=4096,
      maximum_rows=0,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      function_symbol="valid",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("wrong live-row rejection")
  } noraise {
    _ => fail("empty live-row bound was admitted")
  }
  try
    block_control_v1_bf16_f32_source(
      stream_width=4,
      hidden_width=4096,
      maximum_rows=2,
      sinkhorn_iterations=19,
      control_epsilon=-0.000001,
      normalization_epsilon=0.000001,
      function_symbol="not-valid",
    )
  catch {
    InvalidEpsilon => ()
    _ => fail("wrong epsilon rejection precedence")
  } noraise {
    _ => fail("invalid epsilon was admitted")
  }
}

///|
test "pre reduction fixes directional layouts ordered reduction and one BF16 round" {
  let rendered = pre_reduce_v1_bf16_source(
    stream_width=4,
    hidden_width=4096,
    maximum_rows=2,
    function_symbol="lunaflux_hc_pre_reduce_v1",
  )
  assert_eq(rendered.operation(), PreReduceV1)
  assert_eq(rendered.numerical_contract(), OrderedF32PreReduceBf16RoundV1)
  assert_eq(rendered.stream_width(), 4)
  assert_eq(rendered.hidden_width(), 4096)
  assert_eq(rendered.maximum_rows(), 2)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const int32_t live_rows = counts[3]"))
  assert_true(source.contains("const __nv_bfloat16 *streams"))
  assert_true(source.contains("const float *pre"))
  assert_true(source.contains("__nv_bfloat16 *output"))
  assert_true(source.contains("stream < 4"))
  assert_true(source.contains("stream * 4096ULL"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_eq(source.split("__float2bfloat16_rn").length(), 2)
  try
    pre_reduce_v1_bf16_source(
      stream_width=4,
      hidden_width=4096,
      maximum_rows=0,
      function_symbol="valid",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("wrong pre-reduction row-bound rejection")
  } noraise {
    _ => fail("unbounded pre-reduction rows were admitted")
  }
}

///|
test "post combination fixes destination-source order and one BF16 round" {
  let rendered = post_combine_v1_bf16_source(
    stream_width=4,
    hidden_width=4096,
    maximum_rows=2,
    function_symbol="lunaflux_hc_post_combine_v1",
  )
  assert_eq(rendered.operation(), PostCombineV1)
  assert_eq(rendered.numerical_contract(), OrderedF32PostCombineBf16RoundV1)
  assert_eq(rendered.stream_width(), 4)
  assert_eq(rendered.hidden_width(), 4096)
  assert_eq(rendered.maximum_rows(), 2)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const __nv_bfloat16 *branch"))
  assert_true(source.contains("const __nv_bfloat16 *residual"))
  assert_true(source.contains("const float *post"))
  assert_true(source.contains("const float *combination"))
  assert_true(source.contains("destination * 4ULL"))
  assert_true(source.contains("source * 4096ULL"))
  assert_true(source.contains("source < 4"))
  assert_true(source.contains("__fmul_rn(post_value, branch_value)"))
  assert_true(source.contains("__fadd_rn"))
  assert_eq(source.split("__float2bfloat16_rn").length(), 2)
}
