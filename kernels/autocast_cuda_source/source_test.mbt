///|
test "autocast storage lowering preserves math while distinguishing equal-width formats" {
  let template = "#include <cuda_bf16.h>\n__nv_bfloat16 x; float v=__bfloat162float(x); x=__float2bfloat16_rn(__fadd_rn(v, 1.0f));"
  assert_eq(@autocast_cuda_source.storage_source(template, Bfloat16), template)
  assert_eq(
    @autocast_cuda_source.storage_source(template, Float16),
    "#include <cuda_fp16.h>\n__half x; float v=__half2float(x); x=__float2half_rn(__fadd_rn(v, 1.0f));",
  )
  assert_eq(
    @autocast_cuda_source.round_source("sum", Float16),
    "__half2float(__float2half_rn(sum))",
  )
}
