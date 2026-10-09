///|
pub struct LearnedIndexQueriesCudaSource {
  priv plan : @precision.LearnedIndexQueriesPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
  priv transform : @transform.RotaryHadamardFp4CudaSource
}

///|
pub fn learned_index_queries_source(
  plan : @precision.LearnedIndexQueriesPrecision,
  symbol_prefix~ : String,
) -> LearnedIndexQueriesCudaSource raise {
  // Shared renderers validate symbol syntax; no hardware decisions in the plan.
  let (hidden, rank, heads, width, rows) = plan.geometry()
  let symbols = [
    symbol_prefix + "_queries",
    symbol_prefix + "_head_projection",
    symbol_prefix + "_head_scale",
  ]
  let projection = @fp8.block_fp8_ue8m0_projection_cuda_source(
    input_width=rank,
    output_width=heads * width,
    maximum_rows=rows,
    function_symbol=symbols[0],
  )
  let weights = @bf16.dense_projection_bf16_source(
    input_width=hidden,
    output_width=heads,
    maximum_rows=rows,
    function_symbol=symbols[1],
  )
  let transform = @transform.rotary_hadamard_fp4_source(
    plan.transform(),
    symbol_prefix=symbol_prefix + "_transform",
  )
  let scale = @literal.canonical_cuda_float_literal(plan.head_scale())
  let source = projection.source_bytes() +
    weights.source_bytes() +
    transform.source_bytes() +
    @utf8.encode(
      (
        $|extern "C" __global__ void \{symbols[2]}(const int32_t *counts, const __nv_bfloat16 *input, __nv_bfloat16 *output) {
        $|  const int live = counts[3]; if (live <= 0 || live > \{rows}) return;
        $|  const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        $|  if (index >= (unsigned long long)live * \{heads}ULL) return;
        $|  output[index] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(input[index]), \{scale}));
        $|}
        $|
      ),
    )
  { plan, source, symbols: ReadOnlyArray::from_array(symbols), transform, }
}

///|
pub fn LearnedIndexQueriesCudaSource::plan(
  self : Self,
) -> @precision.LearnedIndexQueriesPrecision {
  self.plan
}

///|
pub fn LearnedIndexQueriesCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn LearnedIndexQueriesCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn LearnedIndexQueriesCudaSource::transform(
  self : Self,
) -> @transform.RotaryHadamardFp4CudaSource {
  self.transform
}

///|
pub fn LearnedIndexQueriesCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (_, _, heads, width, rows) = self.plan.geometry()
  if stage == 0 {
    (rows, (heads * width + 255) / 256, 256)
  } else {
    ((rows * heads + 255) / 256, 1, 256)
  }
}
