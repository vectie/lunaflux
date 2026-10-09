///|
pub struct RecurrentBlockCudaSource {
  priv plan : @precision.RecurrentBlockPrecision
  priv prefix : String
  priv source : Bytes
  priv delta : @delta.RecurrentDeltaCudaSource
  priv convolution : @convolution.CausalConvolutionCudaSource
}

///|
/// Per-output parallel projections preserve the ordered numerical contract.
/// These are executable correctness-first schedules, not tensor-core GEMM.
pub fn recurrent_block_source(
  plan : @precision.RecurrentBlockPrecision,
  symbol_prefix~ : String,
) -> RecurrentBlockCudaSource raise {
  // The recurrent renderer validates the prefix as a CUDA identifier.
  let delta = @delta.recurrent_delta_source(
    plan.delta(),
    symbol=symbol_prefix + "_delta",
  )
  let convolution = @convolution.causal_convolution_source(
    plan.convolution(),
    symbol=symbol_prefix + "_conv",
  )
  let (hidden, rank) = plan.projection_geometry()
  let (heads, dimension, _) = plan.delta().geometry()
  let (channels, _) = plan.convolution().geometry()
  let (rows, _, _) = plan.delta().capacity()
  let shapes = [
    (hidden, channels),
    (hidden, channels),
    (hidden, channels),
    (hidden, rank),
    (rank, channels),
    (hidden, heads),
    (hidden, rank),
    (rank, channels),
    (channels, hidden),
  ]
  let indices = [0, 1, 2, 3, 4, 5, 8, 9, 15]
  for (_, output_width) in shapes {
    guard rows.to_int64() * output_width.to_int64() <= 2147483647L * 256L else {
      raise @precision.PrecisionPlanError::NoImplementation
    }
  }
  let mut source = convolution.source_bytes() + b"\n" + delta.source_bytes()
  for index, (input_width, output_width) in shapes {
    source = source +
      b"\n" +
      @projection.dense_projection_bf16_source(
        input_width~,
        output_width~,
        maximum_rows=rows,
        function_symbol=symbol_prefix + "_" + indices[index].to_string(),
      ).source_bytes()
  }
  source = source +
    b"\n" +
    @decay.safe_lower_bound_sigmoid_decay_f32_source(
      head_count=heads,
      head_dimension=dimension,
      lower_bound=plan.delta().log_decay_lower_bound(),
      maximum_rows=rows,
      function_symbol=symbol_prefix + "_7",
    ).source_bytes()
  let epsilon = @literal.canonical_cuda_float_literal(
    plan.normalization_epsilon(),
  )
  let text =
    $|extern "C" __global__ void \{symbol_prefix}_6(const int *counts, const __nv_bfloat16 *input, __nv_bfloat16 *output) {
    $|  const unsigned long long index=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;
    $|  const int rows=counts[3]; if(rows<=0||rows>\{rows}||index>=(unsigned long long)rows*\{heads}ULL) return;
    $|  output[index]=__float2bfloat16_rn(__fdiv_rn(1.0f,__fadd_rn(1.0f,expf(-__bfloat162float(input[index])))));
    $|}
    $|extern "C" __global__ void \{symbol_prefix}_14(const int *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {
    $|  const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,rows=counts[3];
    $|  if(rows<=0||rows>\{rows}||row>=rows||head>=\{heads}) return;
    $|  const unsigned long long base=((unsigned long long)row*\{heads}ULL+head)*\{dimension}ULL;
    $|  __shared__ float inverse;
    $|  if(lane==0) { float sum=0.0f; for(int k=0;k<\{dimension};++k) { float v=__bfloat162float(input[base+k]); sum=__fadd_rn(sum,__fmul_rn(v,v)); } inverse=rsqrtf(__fadd_rn(__fdiv_rn(sum,\{dimension}.0f),\{epsilon})); }
    $|  __syncthreads();
    $|  for(int k=lane;k<\{dimension};k+=128) { float normalized=__fmul_rn(__bfloat162float(input[base+k]),inverse); float weighted=__fmul_rn(__bfloat162float(weight[k]),normalized); float sigmoid=__fdiv_rn(1.0f,__fadd_rn(1.0f,expf(-__bfloat162float(gate[base+k])))); output[base+k]=__float2bfloat16_rn(__fmul_rn(weighted,sigmoid)); }
    $|}
  {
    plan,
    prefix: symbol_prefix,
    source: source + b"\n" + @utf8.encode(text),
    delta,
    convolution,
  }
}

///|
pub fn RecurrentBlockCudaSource::plan(
  self : Self,
) -> @precision.RecurrentBlockPrecision {
  self.plan
}

///|
pub fn RecurrentBlockCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RecurrentBlockCudaSource::delta(
  self : Self,
) -> @delta.RecurrentDeltaCudaSource {
  self.delta
}

///|
pub fn RecurrentBlockCudaSource::convolution(
  self : Self,
) -> @convolution.CausalConvolutionCudaSource {
  self.convolution
}

///|
pub fn RecurrentBlockCudaSource::symbol(self : Self, stage : Int) -> String {
  self.prefix + "_" + stage.to_string()
}
