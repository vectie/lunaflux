///|
async test "inverse arithmetic intervention exports the actual renderer and frequency law" {
  let plan = @precision.RotaryTensorPrecision::new(
    head_count=64,
    head_width=512,
    rotary_width=64,
    maximum_rows=144,
    maximum_positions=256,
    frequency=Base(theta=10000.0),
    inverse=true,
    maximum_workspace_bytes=9437184L,
  )
  let production = @rotary_kv_cuda_source.rotary_tensor_source(
    plan,
    symbol="production_inverse",
  )
  assert_true(production.contains("sine = -sinf(angle)"))
  assert_true(production.contains("__fadd_rn(__fmul_rn(left, cosine)"))
  assert_false(production.contains("__fmaf_rn"))
  let contracted_plan = @precision.RotaryTensorPrecision::new(
    head_count=64,
    head_width=512,
    rotary_width=64,
    maximum_rows=144,
    maximum_positions=256,
    frequency=Base(theta=10000.0),
    inverse=true,
    maximum_workspace_bytes=9437184L,
    product=ComplexFloat32Fma,
  )
  let contracted = @rotary_kv_cuda_source.rotary_tensor_source(
    contracted_plan,
    symbol="contracted_inverse",
  )
  assert_true(
    contracted.contains("__fmaf_rn(left, cosine, -__fmul_rn(right, sine))"),
  )
  assert_true(
    contracted.contains("__fmaf_rn(right, cosine, __fmul_rn(left, sine))"),
  )
  assert_eq(plan.workspace_bytes(), contracted_plan.workspace_bytes())
  if @env.get_env_var("LUNAFLUX_ROTARY_ARITHMETIC_SOURCE") is Some(path) {
    let source = "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
      production +
      "\n" +
      contracted +
      "\n__device__ float generated_frequency(int pair) {\n" +
      @rotary_kv_cuda_source.rotary_frequency_source(plan.frequency(), 64) +
      "\nreturn frequency;\n}\n"
    @fs.write_file(path, source, create_mode=CreateNew)
  }
}
