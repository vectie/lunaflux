///|
test "block FP8 UE8M0 source fixes official block geometry and arithmetic" {
  let rendered = block_fp8_ue8m0_projection_cuda_source(
    input_width=256,
    output_width=128,
    maximum_rows=3,
    function_symbol="lunaflux_block_fp8_test_v1",
  )
  assert_eq(rendered.block_width(), 128)
  let source = @utf8.decode(rendered.source_bytes())
  // CUDA_VERSION belongs to the driver header, not cuda_runtime.h. The
  // generated source must compile standalone or in a runtime-only consumer.
  assert_true(source.starts_with("#include <cuda.h>\n"))
  assert_true(source.contains("CUDA_VERSION < 12080"))
  assert_true(source.contains("__nv_fp8_e8m0"))
  assert_true(source.contains("maximum = 0.0001f"))
  assert_true(source.contains("__fmul_rn(maximum, 0.002232142857142857f)"))
  assert_true(source.contains("bits & 0x7fffffU"))
  assert_true(source.contains("column / 128"))
  assert_true(source.contains("blockIdx.y * 256"))
  assert_true(source.contains("__fadd_rn(block_accumulator"))
  assert_true(source.contains("__float2bfloat16_rn"))
}

///|
test "block FP8 UE8M0 source rejects hostile geometry and symbol" {
  try
    block_fp8_ue8m0_projection_cuda_source(
      input_width=129,
      output_width=128,
      maximum_rows=1,
      function_symbol="valid_symbol",
    )
  catch {
    InvalidGeometry => ()
    _ => fail("wrong geometry failure")
  } noraise {
    _ => fail("unaligned input width admitted")
  }
  try
    block_fp8_ue8m0_projection_cuda_source(
      input_width=128,
      output_width=128,
      maximum_rows=1,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidSymbol => ()
    _ => fail("wrong symbol failure")
  } noraise {
    _ => fail("hostile symbol admitted")
  }
}
