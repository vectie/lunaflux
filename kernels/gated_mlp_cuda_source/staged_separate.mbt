///|
/// counts-dependent separate-weight lowering of the same staged BF16 SwiGLU
/// law. Each row/inner computes gate/up once into a shared product frame;
/// down columns consume that frame instead of repeating the gate/up fold.
pub fn staged_separate_swiglu_bf16_source(
  hidden_size~ : Int,
  intermediate_size~ : Int,
  maximum_rows~ : Int,
  function_prefix~ : String,
) -> Bytes raise GatedMlpCudaSourceError {
  guard hidden_size > 0 && intermediate_size > 0 else { raise InvalidWidth }
  guard maximum_rows > 0 else { raise InvalidRowCount }
  guard maximum_rows.to_int64() * hidden_size.to_int64() <= 2147483647L &&
    maximum_rows.to_int64() * intermediate_size.to_int64() <= 2147483647L else {
    raise DimensionOverflow
  }
  guard valid_symbol(function_prefix) else { raise InvalidFunctionSymbol }
  let h = hidden_size.to_string()
  let m = intermediate_size.to_string()
  let r = maximum_rows.to_string()
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_gate_up(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, __nv_bfloat16 *product) {\n" +
    "const int32_t rows=counts[3]; if(rows<=0 || rows>" +
    r +
    ") return;\n" +
    "const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x; if(i>=(unsigned long long)rows*" +
    m +
    "ULL) return;\n" +
    "const unsigned long long row=i/" +
    m +
    "ULL, inner=i%" +
    m +
    "ULL; float gate_sum=0.0f, up_sum=0.0f;\n" +
    "for(int column=0; column<" +
    h +
    "; ++column) { const float x=__bfloat162float(input[row*" +
    h +
    "ULL+column]); gate_sum=__fadd_rn(gate_sum,__fmul_rn(x,__bfloat162float(gate_weight[inner*" +
    h +
    "ULL+column]))); up_sum=__fadd_rn(up_sum,__fmul_rn(x,__bfloat162float(up_weight[inner*" +
    h +
    "ULL+column]))); }\n" +
    "const __nv_bfloat16 gate=__float2bfloat16_rn(gate_sum), up=__float2bfloat16_rn(up_sum); const float g=__bfloat162float(gate); const __nv_bfloat16 activated=__float2bfloat16_rn(__fdiv_rn(g,__fadd_rn(1.0f,expf(-g)))); product[i]=__float2bfloat16_rn(__fmul_rn(__bfloat162float(activated),__bfloat162float(up))); }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_down(const int32_t *counts, const __nv_bfloat16 *product, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {\n" +
    "const int32_t rows=counts[3]; if(rows<=0 || rows>" +
    r +
    ") return;\n" +
    "const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x; if(i>=(unsigned long long)rows*" +
    h +
    "ULL) return;\n" +
    "const unsigned long long row=i/" +
    h +
    "ULL, out=i%" +
    h +
    "ULL; float sum=0.0f; for(int inner=0; inner<" +
    m +
    "; ++inner) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(product[row*" +
    m +
    "ULL+inner]),__bfloat162float(down_weight[out*" +
    m +
    "ULL+inner]))); output[i]=__float2bfloat16_rn(sum); }\n",
  )
}
