///|
/// Project pre-activated BF16 conditioning [rows,input_width] through BF16
/// weights [groups*fields*width,input_width] and bias. Output is field-major
/// [fields,rows*groups,width], directly consumed by indexed modulation.
/// One thread computes one output; no per-token expansion or transpose launch.
pub fn grouped_parameter_projection_source(
  rows~ : Int,
  input_width~ : Int,
  width~ : Int,
  groups~ : Int,
  fields~ : Int,
  function_symbol~ : String,
) -> Bytes raise AffineModulationCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  if input_width <= 0 || width <= 0 || groups <= 0 || fields <= 0 {
    raise InvalidWidth
  }
  let group_fields = groups.to_int64() * fields.to_int64()
  if group_fields > 2147483647L / width.to_int64() {
    raise DimensionOverflow
  }
  let outputs = group_fields * width.to_int64()
  if rows.to_int64() > 2147483647L / outputs ||
    outputs > 9223372036854775807L / input_width.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let total = rows.to_int64() * outputs
  @utf8.encode(
    "#include <cuda_bf16.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, __nv_bfloat16 *output) {\n" +
    "const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    total.to_string() +
    "ULL) return;\n" +
    "const unsigned long long row = i / " +
    outputs.to_string() +
    "ULL; const unsigned long long column = i % " +
    outputs.to_string() +
    "ULL; float sum = 0.0f;\n" +
    "for (int c = 0; c < " +
    input_width.to_string() +
    "; c += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input[row * " +
    input_width.to_string() +
    "ULL + c]), __bfloat162float(weight[column * " +
    input_width.to_string() +
    "ULL + c])));\n" +
    "const unsigned long long channel = column % " +
    width.to_string() +
    "ULL; const unsigned long long field = (column / " +
    width.to_string() +
    "ULL) % " +
    fields.to_string() +
    "ULL; const unsigned long long group = column / " +
    (fields.to_int64() * width.to_int64()).to_string() +
    "ULL;\n" +
    "const unsigned long long destination = (field * " +
    (rows.to_int64() * groups.to_int64()).to_string() +
    "ULL + row * " +
    groups.to_string() +
    "ULL + group) * " +
    width.to_string() +
    "ULL + channel;\n" +
    "output[destination] = __float2bfloat16_rn(__fadd_rn(sum, __bfloat162float(bias[column]))); }\n",
  )
}
