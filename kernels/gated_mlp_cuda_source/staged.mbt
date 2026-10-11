///|
/// Two-stage packed SwiGLU: prefix_gate_up(input, packed_up_gate, product),
/// then prefix_down(product, down_weight, output), each with 256 threads.
/// Product is BF16 [rows,intermediate]. Materializing it eliminates recomputing
/// both projections once per output channel while preserving ordered sums and
/// BF16 rounding points. All operands must be nonoverlapping during each stage.
pub fn staged_packed_swiglu_bf16_source(
  rows~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  function_prefix~ : String,
  dot_reduction_law? : @precision.DotReductionLaw = OrderedFloat32,
) -> Bytes raise GatedMlpCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  if hidden_size <= 0 || intermediate_size <= 0 {
    raise InvalidWidth
  }
  if intermediate_size > 1073741823 ||
    rows.to_int64() > 2147483647L / hidden_size.to_int64() ||
    rows.to_int64() > 2147483647L / intermediate_size.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_prefix) {
    raise InvalidFunctionSymbol
  }
  let h = hidden_size.to_string()
  let m = intermediate_size.to_string()
  let product_elements = (rows.to_int64() * intermediate_size.to_int64()).to_string()
  let output_elements = (rows.to_int64() * hidden_size.to_int64()).to_string()
  let (up_init, up_update, up_finish) = @reduction.dot_accumulator_source(
    dot_reduction_law,
    accumulator="up_sum",
  )
  let (gate_init, gate_update, gate_finish) = @reduction.dot_accumulator_source(
    dot_reduction_law,
    accumulator="gate_sum",
  )
  let (down_init, down_update, down_finish) = @reduction.dot_accumulator_source(
    dot_reduction_law,
    accumulator="sum",
  )
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_gate_up(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *product) {\n" +
    "const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    product_elements +
    "ULL) return;\n" +
    "const unsigned long long row = i / " +
    m +
    "ULL; const unsigned long long inner = i % " +
    m +
    "ULL;\n" +
    up_init +
    " " +
    gate_init +
    " for (int c = 0; c < " +
    h +
    "; c += 1) {\n" +
    "const float x = __bfloat162float(input[row * " +
    h +
    "ULL + c]);\n" +
    "{ const float product = __fmul_rn(x, __bfloat162float(weight[inner * " +
    h +
    "ULL + c])); " +
    up_update +
    " }\n" +
    "{ const float product = __fmul_rn(x, __bfloat162float(weight[(inner + " +
    m +
    "ULL) * " +
    h +
    "ULL + c])); " +
    gate_update +
    " } }\n" +
    up_finish +
    " " +
    gate_finish +
    "\n" +
    "const __nv_bfloat16 up = __float2bfloat16_rn(up_sum); const __nv_bfloat16 gate = __float2bfloat16_rn(gate_sum);\n" +
    "const float g = __bfloat162float(gate); const __nv_bfloat16 activated = __float2bfloat16_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))));\n" +
    "product[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(up), __bfloat162float(activated))); }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_down(const __nv_bfloat16 *product, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    output_elements +
    "ULL) return;\n" +
    "const unsigned long long row = i / " +
    h +
    "ULL; const unsigned long long out = i % " +
    h +
    "ULL; " +
    down_init +
    "\n" +
    "for (int c = 0; c < " +
    m +
    "; c += 1) { const float product_value = __fmul_rn(__bfloat162float(product[row * " +
    m +
    "ULL + c]), __bfloat162float(weight[out * " +
    m +
    "ULL + c])); const float product = product_value; " +
    down_update +
    " }\n" +
    down_finish +
    "\n" +
    "output[i] = __float2bfloat16_rn(sum); }\n",
  )
}
