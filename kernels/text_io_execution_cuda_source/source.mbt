///|
pub struct TextIoCudaSource {
  priv plan : @precision.TextIoPrecision
  priv prefix : String
  priv bytes : Bytes
  priv order : ReadOnlyArray[Int]
  priv shard : @parallel.VocabularyShard?
}

///|
/// Correctness-first reference lowering. No model family or runtime JIT.
/// Output counts and row selection are device inputs, prepared once by caller.
pub fn text_io_source(
  plan : @precision.TextIoPrecision,
  symbol_prefix~ : String,
  vocabulary_shard? : @parallel.VocabularyShard,
) -> TextIoCudaSource raise @precision.PrecisionPlanError {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 100 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (vocabulary, hidden, _, rows, outputs) = plan.geometry()
  if vocabulary_shard is Some(shard) {
    guard shard.geometry() == (vocabulary, outputs) &&
      plan.head_law() == ShapeAwareFloat32 else {
      raise @precision.PrecisionPlanError::NoImplementation
    }
  }
  let v = vocabulary.to_string()
  let h = hidden.to_string()
  let s = plan.fold(StreamUpdate).extent.to_string()
  let r = rows.to_string()
  let o = outputs.to_string()
  let epsilon = @literal.canonical_cuda_float_literal(
    plan.normalization_epsilon(),
  )
  fn expression(
    op : @precision.TextExpression,
    bindings : ArrayView[String],
  ) -> String raise @precision.PrecisionPlanError {
    @arithmetic.scalar_expression(plan.expression(op), bindings)
    .split(", ")
    .map(part => part.to_owned())
    .collect()
    .join(",")
  }
  fn seed(
    op : @precision.TextExpression,
  ) -> String raise @precision.PrecisionPlanError {
    @arithmetic.scalar_expression(
      @precision.NumericExpression::new(
        [],
        Float32Bits(plan.fold(op).initial_bits),
      ),
      [],
    )
  }
  let stream_seed = seed(StreamUpdate)
  let square_seed = seed(SquareUpdate)
  let head_seed = if plan.head_law() == ScalarFloat32 {
    seed(HeadUpdate)
  } else {
    "0.0f"
  }
  let square_extent = plan.fold(SquareUpdate).extent.to_string()
  let head_extent = hidden.to_string()
  let stream_update = expression(StreamUpdate, [
    "sum",
    "residual[((unsigned long long)input_row*\{s}ULL+stream)*\{h}ULL+c]",
  ])
  let mean = expression(Mean, ["sum", s + ".0f"])
  let square_update = expression(SquareUpdate, ["squares", "mean"])
  let inverse = expression(Inverse, ["squares", h + ".0f", epsilon])
  let normalized = expression(Normalize, ["mean", "inverse"])
  let weight = expression(Weight, ["normalized", "weight[c]"])
  let head_update = expression(HeadUpdate, [
    "sum",
    "hidden[row*\{h}ULL+c]",
    "weight[word*\{h}ULL+c]",
  ])
  let logits = expression(Logits, ["sum"])
  let order = ordered_text_stages(plan.egress_program())
  let prefix_source = "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
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
    "  if(threadIdx.x == 0) { float squares = " +
    square_seed +
    ";\n" +
    "    if(live > 0 && live <= " +
    r +
    " && input_row >= 0 && input_row < live) {\n" +
    "      for(int c=0;c<" +
    square_extent +
    ";++c) { float sum=" +
    stream_seed +
    ";\n" +
    "        for(int stream=0;stream<" +
    s +
    ";++stream) sum=" +
    stream_update +
    ";\n" +
    "        const float mean=" +
    mean +
    "; squares=" +
    square_update +
    ";\n" +
    "      } inverse=" +
    inverse +
    ";\n" +
    "    } else inverse=0.0f;\n  } __syncthreads();\n" +
    "  for(int c=threadIdx.x;c<" +
    h +
    ";c+=256) { float sum=" +
    stream_seed +
    ";\n" +
    "    if(live > 0 && live <= " +
    r +
    " && input_row >= 0 && input_row < live) for(int stream=0;stream<" +
    s +
    ";++stream) sum=" +
    stream_update +
    ";\n" +
    "    const float mean=" +
    mean +
    ";\n" +
    "    const float normalized=" +
    normalized +
    ";\n" +
    "    out[(unsigned long long)row*" +
    h +
    "ULL+c]=" +
    weight +
    ";\n  }\n}\n"
  let ordered_head = "extern \"C\" __global__ void " +
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
    "ULL; float sum=" +
    head_seed +
    ";\n" +
    "  for(int c=0;c<" +
    head_extent +
    ";++c) sum=" +
    head_update +
    ";\n" +
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
  let source = if plan.head_law() == ShapeAwareFloat32 {
    prefix_source +
    parallel_head_source(plan, symbol_prefix, vocabulary_shard?) +
    (match vocabulary_shard {
      None => parallel_greedy_source(plan, symbol_prefix)
      Some(shard) => sharded_greedy_source(plan, shard, symbol_prefix)
    })
  } else {
    prefix_source + ordered_head
  }
  {
    plan,
    prefix: symbol_prefix,
    bytes: @utf8.encode(source),
    order,
    shard: vocabulary_shard,
  }
}

///|
pub fn TextIoCudaSource::head_launch(self : Self) -> (Int, Int) {
  let (vocabulary, _, _, _, outputs) = self.plan.geometry()
  if self.plan.head_law() == ShapeAwareFloat32 {
    ((self.shard.map(s => s.columns().1).unwrap_or(vocabulary) + 3) / 4, 128)
  } else {
    (((outputs.to_int64() * vocabulary.to_int64() + 255L) / 256L).to_int(), 256)
  }
}

///|
pub fn TextIoCudaSource::greedy_threads(self : Self) -> Int {
  if self.plan.head_law() == ShapeAwareFloat32 {
    256
  } else {
    1
  }
}

///|
fn ordered_text_stages(
  program : @parallel.Program,
) -> ReadOnlyArray[Int] raise @precision.PrecisionPlanError {
  program
  .lower(Ordered)
  .commands(ComputeLane)
  .map(command => {
    match command {
      Execute(step~) =>
        match program.steps()[step].action() {
          Compute(stage) => stage
          _ => raise @precision.PrecisionPlanError::NoImplementation
        }
      _ => raise @precision.PrecisionPlanError::NoImplementation
    }
  })
}

///|
pub fn TextIoCudaSource::egress_order(self : Self) -> ReadOnlyArray[Int] {
  self.order
}

///|
pub fn learned_egress_order(
  plan : @precision.LearnedTextIoPrecision,
) -> ReadOnlyArray[Int] raise @precision.PrecisionPlanError {
  ordered_text_stages(plan.egress_program())
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
    4 if self.shard is Some(_) => "greedy_join"
    _ => raise IllegalPlacement
  }
  self.prefix + "_" + suffix
}
