///|
pub(all) suberror Bf16BinaryCudaSourceError {
  InvalidWidth
  InvalidMaximumRows
  InvalidFunctionSymbol
  DimensionOverflow
} derive(Debug, Eq)

///|
pub struct Bf16BinaryCudaSource {
  priv width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
pub fn bf16_add_source(
  width~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> Bf16BinaryCudaSource raise Bf16BinaryCudaSourceError {
  if width <= 0 {
    raise InvalidWidth
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  let rows = maximum_rows.to_int64()
  let columns = width.to_int64()
  if columns > 9223372036854775552L / rows {
    raise DimensionOverflow
  }
  if function_symbol.is_empty() ||
    function_symbol.length() > 128 ||
    !(function_symbol[0] is ('a'..='z' | 'A'..='Z' | '_')) ||
    !function_symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *left, const __nv_bfloat16 *right, __nv_bfloat16 *output) {\n" +
    "  if (output == left || output == right) return; const int32_t live_rows = counts[3]; if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return; const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long live = (unsigned long long)live_rows * " +
    width.to_string() +
    "ULL; if (index >= live) return; output[index] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(left[index]), __bfloat162float(right[index])));\n}\n",
  )
  { width, maximum_rows, function_symbol, source, }
}

///|
pub fn Bf16BinaryCudaSource::width(self : Self) -> Int {
  self.width
}

///|
pub fn Bf16BinaryCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn Bf16BinaryCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn Bf16BinaryCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
