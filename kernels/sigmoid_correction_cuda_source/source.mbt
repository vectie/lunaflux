///|
fn valid_sigmoid_correction_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn sigmoid_and_correction_f32_source(
  expert_count~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> SigmoidCorrectionCudaSource raise SigmoidCorrectionCudaSourceError {
  if expert_count <= 0 || expert_count > 1048576 {
    raise InvalidExpertCount
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_sigmoid_correction_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let maximum_cells = maximum_rows.to_int64() * expert_count.to_int64()
  if maximum_cells <= 0L || maximum_cells > 9223372036854775552L {
    raise DimensionOverflow
  }
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *raw_logits, const float *correction_bias, float *scores, float *scores_for_choice) {\n" +
    "  if (scores == scores_for_choice) return;\n" +
    "  const int row_count = counts[3]; if (row_count <= 0 || row_count > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; const unsigned long long live = (unsigned long long)row_count * " +
    expert_count.to_string() +
    "ULL; if (index >= live) return;\n" +
    "  const unsigned long long expert = index % " +
    expert_count.to_string() +
    "ULL; const float sigmoid_score = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-raw_logits[index])));\n" +
    "  scores[index] = sigmoid_score; scores_for_choice[index] = __fadd_rn(sigmoid_score, correction_bias[expert]);\n}\n",
  )
  {
    operation: SigmoidAndCorrectionF32,
    numerical_contract: Float32SigmoidThenCorrectionAddV1,
    expert_count,
    maximum_rows,
    function_symbol,
    source,
  }
}
