///|
test "KV and standalone cache lower the same explicit complex rounding law" {
  let plan = @precision.RotaryKvPrecision::new(
    query_heads=2,
    key_value_heads=1,
    head_width=512,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=256,
    frequency=Base(theta=10000.0),
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes=9216L,
    product=ComplexFloat32Fma,
  )
  assert_eq(plan.product_law(), @precision.RotaryProductLaw::ComplexFloat32Fma)
  let kv = @rotary_kv_cuda_source.rotary_kv_source(plan, symbol_prefix="pair")
  let cache_plan = @precision.RotaryCachePrecision::new(
    head_count=1,
    head_width=512,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=256,
    frequency=Base(theta=10000.0),
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes=3072L,
    product=ComplexFloat32Fma,
  )
  assert_eq(cache_plan.rotary().product_law(), plan.product_law())
  let cache = @rotary_kv_cuda_source.rotary_cache_source(
    cache_plan,
    symbol_prefix="compact",
  )
  for source in [kv.source_bytes(), cache.source_bytes()] {
    let text = @utf8.decode(source)
    assert_true(
      text.contains("__fmaf_rn(left, cosine, -(__fmul_rn(right, sine)))"),
    )
    assert_true(
      text.contains("__fmaf_rn(right, cosine, __fmul_rn(left, sine))"),
    )
  }
}

///|
test "omitted and explicit separate rounding render identical default source" {
  let omitted = @precision.RotaryTensorPrecision::new(
    head_count=2,
    head_width=512,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=256,
    frequency=Base(theta=10000.0),
    inverse=false,
    maximum_workspace_bytes=6144L,
  )
  let explicit = @precision.RotaryTensorPrecision::new(
    head_count=2,
    head_width=512,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=256,
    frequency=Base(theta=10000.0),
    inverse=false,
    maximum_workspace_bytes=6144L,
    product=SeparateFloat32Products,
  )
  assert_eq(
    omitted.product_law(),
    @precision.RotaryProductLaw::SeparateFloat32Products,
  )
  assert_eq(
    @rotary_kv_cuda_source.rotary_tensor_source(omitted, symbol="same"),
    @rotary_kv_cuda_source.rotary_tensor_source(explicit, symbol="same"),
  )
}
