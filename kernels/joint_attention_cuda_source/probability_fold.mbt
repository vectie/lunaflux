///|
// Pure device rendering of the explicitly selected probability partition.
// Document boundaries, Q/K ownership and all other block operations are
// unchanged. Online storage is numerically distinct from global storage.
fn probability_fold(
  inner : Int,
  dimension : Int,
  scale : Double,
  key_start : String,
  key_end : String,
  law : @precision.AttentionProbabilityLaw,
  probability_value : String,
) -> String {
  if law is OnlineBf16(block_tokens~) {
    let dot = "float dot = 0.0f; const unsigned long long kb = (unsigned long long)kr * \{inner}ULL + (unsigned long long)head * \{dimension}ULL; for (int c = 0; c < \{dimension}; c += 1) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[qb + c]), __bfloat162float(key[kb + c])));"
    return " float maximum = -INFINITY; float denominator = 0.0f; float numerator = 0.0f; for (int start = \{key_start}; start < \{key_end};) { const int end = start + min(\{block_tokens}, \{key_end} - start); float next_maximum = maximum; for (int kr = start; kr < end; kr += 1) { " +
      dot +
      " next_maximum = fmaxf(next_maximum, __fmul_rn(dot, \{scale}f)); } const float alpha = expf(__fsub_rn(maximum, next_maximum)); denominator = __fmul_rn(denominator, alpha); numerator = __fmul_rn(numerator, alpha); float tile_denominator = 0.0f; float tile_numerator = 0.0f; for (int kr = start; kr < end; kr += 1) { " +
      dot +
      " const float probability = expf(__fsub_rn(__fmul_rn(dot, \{scale}f), next_maximum)); tile_denominator = __fadd_rn(tile_denominator, probability); const unsigned long long vi = (unsigned long long)kr * \{inner}ULL + (unsigned long long)head * \{dimension}ULL + vc; tile_numerator = __fadd_rn(tile_numerator, __fmul_rn(__bfloat162float(__float2bfloat16_rn(probability)), __bfloat162float(value[vi]))); } denominator = __fadd_rn(denominator, tile_denominator); numerator = __fadd_rn(numerator, tile_numerator); maximum = next_maximum; start = end; } attention_output[i] = __float2bfloat16_rn(__fdiv_rn(numerator, denominator)); }\n"
  }
  // Keep the existing global translation unit byte-identical.
  " float maximum = -INFINITY; for (int kr = " +
  key_start +
  "; kr < " +
  key_end +
  "; kr += 1) { float dot = 0.0f; const unsigned long long kb = (unsigned long long)kr * " +
  inner.to_string() +
  "ULL + (unsigned long long)head * " +
  dimension.to_string() +
  "ULL; for (int c = 0; c < " +
  dimension.to_string() +
  "; c += 1) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[qb + c]), __bfloat162float(key[kb + c]))); maximum = fmaxf(maximum, __fmul_rn(dot, " +
  scale.to_string() +
  "f)); } float denominator = 0.0f; float numerator = 0.0f; for (int kr = " +
  key_start +
  "; kr < " +
  key_end +
  "; kr += 1) { float dot = 0.0f; const unsigned long long kb = (unsigned long long)kr * " +
  inner.to_string() +
  "ULL + (unsigned long long)head * " +
  dimension.to_string() +
  "ULL; for (int c = 0; c < " +
  dimension.to_string() +
  "; c += 1) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[qb + c]), __bfloat162float(key[kb + c]))); const float probability = expf(__fsub_rn(__fmul_rn(dot, " +
  scale.to_string() +
  "f), maximum)); denominator = __fadd_rn(denominator, probability); const unsigned long long vi = (unsigned long long)kr * " +
  inner.to_string() +
  "ULL + (unsigned long long)head * " +
  dimension.to_string() +
  "ULL + vc; numerator = __fadd_rn(numerator, __fmul_rn(" +
  probability_value +
  ", __bfloat162float(value[vi]))); } attention_output[i] = __float2bfloat16_rn(__fdiv_rn(numerator, denominator)); }\n"
}
