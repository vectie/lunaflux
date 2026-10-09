///|
fn valid_symbol(symbol : StringView) -> Bool {
  if symbol.length() == 0 || symbol.length() > 128 {
    return false
  }
  for index, code in symbol {
    if index == 0 {
      if !((code >= 'A' && code <= 'Z') ||
        (code >= 'a' && code <= 'z') ||
        code == '_') {
        return false
      }
    } else if !((code >= '0' && code <= '9') ||
      (code >= 'A' && code <= 'Z') ||
      (code >= 'a' && code <= 'z') ||
      code == '_') {
      return false
    }
  }
  true
}

///|
pub fn f32_residual_and_three_way_average_source(
  element_count~ : Int64,
  residual_symbol~ : String,
  average_symbol~ : String,
) -> JointAudioAmpCudaSource raise JointAudioAmpCudaSourceError {
  if element_count <= 0L {
    raise InvalidElementCount
  }
  if !valid_symbol(residual_symbol) ||
    !valid_symbol(average_symbol) ||
    residual_symbol == average_symbol {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <stdint.h>\nextern \"C\" __global__ void " +
    residual_symbol +
    "(const float *residual, const float *value, float *output) { const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i < " +
    element_count.to_string() +
    "ULL) output[i] = __fadd_rn(residual[i], value[i]); }\n" +
    "extern \"C\" __global__ void " +
    average_symbol +
    "(const float *block0, const float *block1, const float *block2, float *output) { const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i < " +
    element_count.to_string() +
    "ULL) { const float sum01 = __fadd_rn(block0[i], block1[i]); const float sum012 = __fadd_rn(sum01, block2[i]); output[i] = __fdiv_rn(sum012, 3.0f); }}\n",
  )
  {
    numerical_contract: OrderedF32ResidualAndThreeWayAverageV1,
    element_count,
    residual_symbol,
    average_symbol,
    source,
  }
}
