///|
pub struct TextIoCudaSource {
  priv plan : @precision.TextIoPrecision
  priv prefix : String
  priv bytes : Bytes
}

///|
/// Correctness-first reference lowering. No model family or runtime JIT.
/// Output counts and row selection are device inputs, prepared once by caller.
pub fn text_io_source(
  plan : @precision.TextIoPrecision,
  symbol_prefix~ : String,
) -> TextIoCudaSource raise @precision.PrecisionPlanError {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 100 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise IllegalPlacement
  }
  let (vocabulary, hidden, streams, rows, outputs) = plan.geometry()
  let v = vocabulary.to_string()
  let h = hidden.to_string()
  let s = streams.to_string()
  let r = rows.to_string()
  let o = outputs.to_string()
  let epsilon = @literal.canonical_cuda_float_literal(
    plan.normalization_epsilon(),
  )
  let normalized = match plan.normalization_law() {
    Bf16NormalizeThenWeight =>
      "__bfloat162float(__float2bfloat16_rn(__fmul_rn(mean,inverse)))"
    F32NormalizeThenWeight => "__fmul_rn(mean,inverse)"
  }
  let logits = match plan.logits_law() {
    Bf16Round => "__bfloat162float(__float2bfloat16_rn(sum))"
    F32Accumulate => "sum"
  }
  let source = "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    symbol_prefix +
    "_embed(" +
    "const int32_t* counts, const int32_t* ids, const __nv_bfloat16* table, __nv_bfloat16* out) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x*256ULL+threadIdx.x;\n" +
    "  const int live = counts[3]; if(live <= 0 || live > " +
    r +
    ") return;\n" +
    "  if(i >= (unsigned long long)live*" +
    s +
    "ULL*" +
    h +
    "ULL) return;\n" +
    "  const int token = ids[i/(" +
    s +
    "ULL*" +
    h +
    "ULL)];\n" +
    "  out[i] = token >= 0 && token < " +
    v +
    " ? table[(unsigned long long)token*" +
    h +
    "ULL+i%" +
    h +
    "ULL] : __float2bfloat16_rn(0.0f);\n}\n" +
    "extern \"C\" __global__ void " +
    symbol_prefix +
    "_norm(" +
    "const int32_t* counts, const int32_t* output_count, const int32_t* selected, const __nv_bfloat16* residual, const __nv_bfloat16* weight, __nv_bfloat16* out) {\n" +
    "  const int row = blockIdx.x; const int n = output_count[0];\n" +
    "  if(n <= 0 || n > " +
    o +
    " || row >= n) return;\n" +
    "  const int input_row = selected[row]; const int live = counts[3];\n" +
    "  __shared__ float inverse;\n" +
    "  if(threadIdx.x == 0) { float squares = 0.0f;\n" +
    "    if(live > 0 && live <= " +
    r +
    " && input_row >= 0 && input_row < live) {\n" +
    "      for(int c=0;c<" +
    h +
    ";++c) { float sum=0.0f;\n" +
    "        for(int stream=0;stream<" +
    s +
    ";++stream) sum=__fadd_rn(sum,__bfloat162float(residual[((unsigned long long)input_row*" +
    s +
    "ULL+stream)*" +
    h +
    "ULL+c]));\n" +
    "        const float mean=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(sum," +
    s +
    ".0f))); squares=__fadd_rn(squares,__fmul_rn(mean,mean));\n" +
    "      } inverse=rsqrtf(__fadd_rn(__fdiv_rn(squares," +
    h +
    ".0f)," +
    epsilon +
    "));\n" +
    "    } else inverse=0.0f;\n  } __syncthreads();\n" +
    "  for(int c=threadIdx.x;c<" +
    h +
    ";c+=256) { float sum=0.0f;\n" +
    "    if(live > 0 && live <= " +
    r +
    " && input_row >= 0 && input_row < live) for(int stream=0;stream<" +
    s +
    ";++stream) sum=__fadd_rn(sum,__bfloat162float(residual[((unsigned long long)input_row*" +
    s +
    "ULL+stream)*" +
    h +
    "ULL+c]));\n" +
    "    const float mean=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(sum," +
    s +
    ".0f)));\n" +
    "    const float normalized=" +
    normalized +
    ";\n" +
    "    out[(unsigned long long)row*" +
    h +
    "ULL+c]=__float2bfloat16_rn(__fmul_rn(normalized,__bfloat162float(weight[c])));\n  }\n}\n" +
    "extern \"C\" __global__ void " +
    symbol_prefix +
    "_head(" +
    "const int32_t* output_count, const __nv_bfloat16* hidden, const __nv_bfloat16* weight, float* logits) {\n" +
    "  const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x; const int n=output_count[0];\n" +
    "  if(n<=0 || n>" +
    o +
    " || i>=(unsigned long long)n*" +
    v +
    "ULL) return;\n" +
    "  const unsigned long long row=i/" +
    v +
    "ULL, word=i%" +
    v +
    "ULL; float sum=0.0f;\n" +
    "  for(int c=0;c<" +
    h +
    ";++c) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(hidden[row*" +
    h +
    "ULL+c]),__bfloat162float(weight[word*" +
    h +
    "ULL+c])));\n" +
    "  logits[i]=" +
    logits +
    ";\n}\n" +
    "extern \"C\" __global__ void " +
    symbol_prefix +
    "_greedy(" +
    "const int32_t* counts, const int32_t* output_count, const int32_t* selected, const float* logits, int32_t* tokens) {\n" +
    "  const int row=blockIdx.x; const int n=output_count[0]; if(threadIdx.x!=0 || n<=0 || n>" +
    o +
    " || row>=n) return;\n" +
    "  if(counts[3]<=0 || counts[3]>" +
    r +
    " || selected[row]<0 || selected[row]>=counts[3]) { tokens[row]=-1; return; }\n" +
    "  float best=-INFINITY; int token=-1; for(int word=0;word<" +
    v +
    ";++word) { const float value=logits[(unsigned long long)row*" +
    v +
    "ULL+word]; if(!isfinite(value)) { tokens[row]=-1; return; } if(token<0 || value>best) {best=value;token=word;} } tokens[row]=token;\n}\n"
  { plan, prefix: symbol_prefix, bytes: @utf8.encode(source), }
}

///|
pub fn TextIoCudaSource::plan(self : Self) -> @precision.TextIoPrecision {
  self.plan
}

///|
pub fn TextIoCudaSource::source_bytes(self : Self) -> Bytes {
  self.bytes
}

///|
pub fn TextIoCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String raise @precision.PrecisionPlanError {
  let suffix = match stage {
    0 => "embed"
    1 => "norm"
    2 => "head"
    3 => "greedy"
    _ => raise IllegalPlacement
  }
  self.prefix + "_" + suffix
}
