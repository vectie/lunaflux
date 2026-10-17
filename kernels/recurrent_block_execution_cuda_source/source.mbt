///|
pub struct RecurrentBlockCudaSource {
  priv plan : @precision.RecurrentBlockPrecision
  priv prefix : String
  priv source : Bytes
  priv delta : @delta.RecurrentDeltaCudaSource
  priv convolution : @convolution.CausalConvolutionCudaSource
  priv input_region : @matrix.SegmentedMatrixPlan?
}

///|
/// Per-output parallel projections preserve the ordered numerical contract.
/// These are executable correctness-first schedules, not tensor-core GEMM.
pub fn recurrent_block_source(
  plan : @precision.RecurrentBlockPrecision,
  symbol_prefix~ : String,
  fuse_input_projections? : Bool = false,
  matrix_staging? : @matrix.MatrixStaging = Synchronous,
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
  let input_region = if fuse_input_projections {
    Some(
      @matrix.SegmentedMatrixPlan::new(
        [channels, channels, channels, rank, heads, rank].map(columns => {
          @matrix.MatrixTilePlan::new(
            rows~,
            columns~,
            inner=hidden,
            law=plan.projection_law(),
            staging=matrix_staging,
          )
        }),
      ),
    )
  } else {
    None
  }
  for (_, output_width) in shapes {
    guard rows.to_int64() * output_width.to_int64() <= 2147483647L * 256L else {
      raise @precision.PrecisionPlanError::NoImplementation
    }
  }
  let mut source = convolution.source_bytes() + b"\n" + delta.source_bytes()
  for index, (input_width, output_width) in shapes {
    if input_region is Some(_) && indices[index] is (0 | 1 | 2 | 3 | 5 | 8) {
      continue
    }
    let symbol = symbol_prefix + "_" + indices[index].to_string()
    let partial = indices[index] == 15 &&
      plan.publication() == ReduceFloat32BeforeRound
    let projection = if plan.projection_law() != ScalarFloat32 {
      @tiled.dense_source(
        @matrix.MatrixTilePlan::new(
          rows~,
          columns=output_width,
          inner=input_width,
          law=plan.projection_law(),
          staging=matrix_staging,
        ),
        symbol~,
        partial_float32=partial,
      )
    } else if partial {
      @tiled.scalar_partial_source(
        rows~,
        columns=output_width,
        inner=input_width,
        symbol~,
      )
    } else {
      @projection.dense_projection_bf16_source(
        input_width~,
        output_width~,
        maximum_rows=rows,
        function_symbol=symbol,
      ).source_bytes()
    }
    source = source + b"\n" + projection
  }
  if input_region is Some(region) {
    source = source +
      b"\n" +
      @tiled.segmented_source(region, symbol=symbol_prefix + "_inputs")
  }
  if plan.publication() == ReduceFloat32BeforeRound {
    source = source +
      b"\n" +
      @tiled.publication_source(
        rows~,
        width=hidden,
        symbol=symbol_prefix + "_16",
      )
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
  let beta = @arithmetic.scalar_expression(
    @precision.RecurrentBlockExpression::Beta.program(),
    ["input[index]"],
  )
  let square = @arithmetic.scalar_expression(
    @precision.RmsExpression::SquareUpdate.program(),
    ["sum", "v"],
  )
  let variance = @arithmetic.scalar_expression(
    @precision.RmsExpression::Variance.program(),
    ["sum", "\{dimension}.0f", epsilon],
  )
  let inverse = @arithmetic.scalar_expression(
    @precision.ReciprocalRootLaw::ReciprocalSqrtFloat32.expression(),
    [variance],
  )
  let output_gate = @arithmetic.scalar_expression(
    @precision.RecurrentBlockExpression::OutputGate.program(),
    ["input[base+k]", "inverse", "weight[k]", "gate[base+k]"],
  )
  let text =
    $|extern "C" __global__ void \{symbol_prefix}_6(const int *counts, const __nv_bfloat16 *input, __nv_bfloat16 *output) {
    $|  const unsigned long long index=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;
    $|  const int rows=counts[3]; if(rows<=0||rows>\{rows}||index>=(unsigned long long)rows*\{heads}ULL) return;
    $|  output[index]=\{beta};
    $|}
    $|extern "C" __global__ void \{symbol_prefix}_14(const int *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {
    $|  const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,rows=counts[3];
    $|  if(rows<=0||rows>\{rows}||row>=rows||head>=\{heads}) return;
    $|  const unsigned long long base=((unsigned long long)row*\{heads}ULL+head)*\{dimension}ULL;
    $|  __shared__ float inverse;
    $|  if(lane==0) { float sum=0.0f; for(int k=0;k<\{dimension};++k) { float v=__bfloat162float(input[base+k]); sum=\{square}; } inverse=\{inverse}; }
    $|  __syncthreads();
    $|  for(int k=lane;k<\{dimension};k+=128) { output[base+k]=\{output_gate}; }
    $|}
  {
    plan,
    prefix: symbol_prefix,
    source: source + b"\n" + @utf8.encode(text),
    delta,
    convolution,
    input_region,
  }
}

///|
pub fn RecurrentBlockCudaSource::projection_launch(
  self : Self,
  input_width : Int,
  output_width : Int,
) -> (Int, Int) raise {
  let (rows, _, _) = self.plan.delta().capacity()
  if self.plan.projection_law() != ScalarFloat32 {
    let tile = @matrix.MatrixTilePlan::new(
      rows~,
      columns=output_width,
      inner=input_width,
      law=self.plan.projection_law(),
    )
    (tile.tile_count(), 128)
  } else {
    (((rows.to_int64() * output_width.to_int64() + 255L) / 256L).to_int(), 256)
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
  self.prefix + "_" + (if stage == 17 { "inputs" } else { stage.to_string() })
}

///|
pub fn RecurrentBlockCudaSource::input_region_tiles(self : Self) -> Int? {
  self.input_region.map(region => region.tile_count())
}
