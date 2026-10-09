///|
/// Prepare shared conditioning once per denoising evaluation, before all
/// block/final AdaLN projections: F32 SiLU, then BF16 cast. Input must not have
/// been pre-cast to BF16. One element per thread, 256 threads per block.
pub fn f32_silu_to_bf16_source(
  elements~ : Int64,
  function_symbol~ : String,
) -> Bytes raise AffineModulationCudaSourceError {
  if elements <= 0L || elements > 2147483647L {
    raise InvalidRowCount
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const float *input, __nv_bfloat16 *output) {\n" +
    "const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    elements.to_string() +
    "ULL) return;\n" +
    "const float x = input[i]; const float activated = __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));\n" +
    "output[i] = __float2bfloat16_rn(activated); }\n",
  )
}
