///|
/// The extra region stores a bounded physical grouping, not model activations.
/// It is allocated once with the activation frame and survives gate/down.
pub fn PackedExpertMlpCudaSource::activation_bytes(self : Self) -> Int64 {
  let bytes = self.plan.workspace_bytes().0
  if self.plan.dot_law() == ScalarFloat32 {
    return bytes
  }
  let (rows, _, _, _, selected) = self.plan.geometry()
  (bytes + 3L) / 4L * 4L +
  (1L + rows.to_int64() * selected.to_int64() * 17L) * 4L
}

///|
fn grouped_expert_source(
  plan : @precision_ir.ExpertMlpPrecision,
  symbol~ : String,
) -> PackedExpertMlpCudaSource raise {
  guard plan.activation_quantization() is None &&
    plan.scaled_dot_law() == DecodeOperandsFloat32 else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let (rows, hidden, intermediate, experts, selected) = plan.geometry()
  let grouping = @matrix.ExpertGroupingPlan::new(
    maximum_rows=rows,
    selected_experts=selected,
  )
  let jobs = grouping.maximum_groups()
  let metadata = (plan.workspace_bytes().0 + 3L) / 4L * 4L
  let gate_plan = @matrix.MatrixTilePlan::new(
    rows=16,
    columns=intermediate,
    inner=hidden,
    law=plan.dot_law(),
  )
  let down_plan = @matrix.MatrixTilePlan::new(
    rows=16,
    columns=hidden,
    inner=intermediate,
    law=plan.dot_law(),
  )
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=32,
    vector=1,
    rotations=1,
    row_group=1,
  ).unwrap()
  let region = @physical_tile_ir.OperandStorage::new(
      layout,
      [1],
      owners=32,
      stages=1,
    )
    .unwrap()
    .region(0)
    .unwrap()
  let loads : Array[String] = []
  let addresses : Array[String] = []
  for i, weight in plan.weights() {
    loads.push(
      @luna_cuda_precision.tile_load_source(
        weight.refine_region(region),
        symbol="\{symbol}_w\{i}",
      ),
    )
    let storage = @precision_ir.packed_buffer_layout(weight.representation())
    let offset = plan.weight_offsets()[i]
    let scales = storage.scales_offset().unwrap_or(0L)
    let global = storage.global_scale_offset().unwrap_or(0L)
    addresses.push(
      "const unsigned char *w\{i}=bank+(unsigned long long)(local<0?0:local)*\{plan.expert_stride()}ULL+\{offset}ULL; const unsigned char *s\{i}=w\{i}+\{scales}ULL; const float *g\{i}=(const float *)(w\{i}+\{global}ULL);",
    )
  }
  let input = "(job_ids[matrix_row]>=0&&local>=0?input[(unsigned long long)(job_ids[matrix_row]/\{selected})*\{hidden}+matrix_inner]:__float2bfloat16_rn(0.0f))"
  let bodies : Array[String] = []
  for i in 0..<2 {
    bodies.push(
      @tiled.matrix_tile_body(
        gate_plan,
        input~,
        weight="(local>=0&&column_base+matrix_column<\{intermediate}?\{symbol}_w\{i}(w\{i},s\{i},g\{i},column_base+matrix_column,matrix_inner):__float2bfloat16_rn(0.0f))",
        result="result\{i}",
      ),
    )
  }
  let down_body = @tiled.matrix_tile_body(
    down_plan,
    input="(job_ids[matrix_row]>=0&&local>=0?activation[(unsigned long long)job_ids[matrix_row]*\{intermediate}+matrix_inner]:__float2bfloat16_rn(0.0f))",
    weight="(local>=0&&column_base+matrix_column<\{hidden}?\{symbol}_w2(w2,s2,g2,column_base+matrix_column,matrix_inner):__float2bfloat16_rn(0.0f))",
    result="result",
  )
  fn expr(
    op : @precision_ir.ExpertExpression,
    values : ArrayView[String],
  ) -> String raise {
    @luna_cuda_precision.scalar_expression(plan.expression(op), values)
  }
  let clamp = match plan.clamp() {
    None => ""
    Some(limit) => {
      let literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(limit)
      "gate=\{expr(ClampGate, ["gate", literal])}; up=\{expr(ClampUp, ["up", literal])};"
    }
  }
  let before = plan.routing_placement() == BeforeDownInputRound
  let routing_arg = if before { ",const float *selected_weights" } else { "" }
  let route = if before {
    "product=\{expr(ApplyRoute, ["product", "selected_weights[job]"])};"
  } else {
    ""
  }
  let product = expr(Product, ["g", "up"])
  let publish = expr(DownPublish, ["result[i]", "selected_weights[job]"])
  let combine = expr(CombineUpdate, [
    "sum",
    "weighted[(row*\{selected}ULL+slot)*\{hidden}ULL+column]",
  ])
  let group_symbol = symbol + "_group"
  let gate_symbol = symbol + "_gate_up"
  let down_symbol = symbol + "_down"
  let combine_symbol = symbol + "_combine"
  let n_gate = (intermediate + 63) / 64
  let n_down = (hidden + 63) / 64
  guard jobs.to_int64() * n_gate.to_int64() <= 2147483647L &&
    jobs.to_int64() * n_down.to_int64() <= 2147483647L else {
    raise @precision_ir.PrecisionPlanError::ArithmeticOverflow
  }
  let text = "#include <cuda_bf16.h>\n#include <mma.h>\n" +
    loads.join("\n") +
    (
      $|
      $|extern "C" __global__ void \{group_symbol}(const int *counts,const int *selected_indices,__nv_bfloat16 *activation) {
      $|  int *meta=(int *)((unsigned char *)activation+\{metadata}ULL);
      $|  for(int i=threadIdx.x;i<\{1+jobs*17};i+=128) meta[i]=-1;
      $|  __syncthreads();
      $|  if(threadIdx.x!=0) return;
      $|  meta[0]=0; const int rows=counts[3]; if(rows<=0||rows>\{rows}) return;
      $|  const int live=rows*\{selected}; int groups=0;
      $|  for(int first=0;first<live;++first) {
      $|    int expert=selected_indices[first]; bool seen=false;
      $|    for(int j=0;j<first;++j) if(selected_indices[j]==expert) {seen=true;break;}
      $|    if(seen) continue;
      $|    int matched=0;
      $|    for(int j=first;j<live;++j) if(selected_indices[j]==expert) {
      $|      int group=groups+matched/16; meta[1+group]=expert;
      $|      meta[1+\{jobs}+group*16+matched%16]=j; ++matched;
      $|    }
      $|    groups+=(matched+15)/16;
      $|  }
      $|  meta[0]=groups;
      $|}
      $|extern "C" __global__ void \{gate_symbol}(const int *counts,const __nv_bfloat16 *input,const int *selected_indices,const int *expert_map,const unsigned char *bank,__nv_bfloat16 *activation\{routing_arg}) {
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);
      $|  const int group=blockIdx.x/\{n_gate},column_base=(blockIdx.x%\{n_gate})*64;
      $|  if(group>=meta[0]) return;
      $|  const int expert=meta[1+group]; int local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;
      $|  if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{addresses[0]}
      $|  \{addresses[1]}
      $|  __shared__ __align__(32) float result0[1024],result1[1024];
      $|  \{bodies[0]}
      $|  \{bodies[1]}
      $|  for(int i=threadIdx.x;i<1024;i+=128) {
      $|    int job=job_ids[i/64],column=column_base+i%64;
      $|    if(job<0||column>=\{intermediate}) continue;
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(result0[i]),up=__float2bfloat16_rn(result1[i]); \{clamp}
      $|    float g=__bfloat162float(gate),product=\{product}; \{route}
      $|    activation[(unsigned long long)job*\{intermediate}+column]=local<0?__float2bfloat16_rn(0.0f):__float2bfloat16_rn(product);
      $|  }
      $|}
      $|extern "C" __global__ void \{down_symbol}(const int *counts,const __nv_bfloat16 *activation,const int *selected_indices,const float *selected_weights,const int *expert_map,const unsigned char *bank,__nv_bfloat16 *weighted) {
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);
      $|  const int group=blockIdx.x/\{n_down},column_base=(blockIdx.x%\{n_down})*64;
      $|  if(group>=meta[0]) return;
      $|  const int expert=meta[1+group]; int local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;
      $|  if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{addresses[2]}
      $|  __shared__ __align__(32) float result[1024];
      $|  \{down_body}
      $|  for(int i=threadIdx.x;i<1024;i+=128) {
      $|    int job=job_ids[i/64],column=column_base+i%64;
      $|    if(job>=0&&column<\{hidden}) weighted[(unsigned long long)job*\{hidden}+column]=local<0?__float2bfloat16_rn(0.0f):\{publish};
      $|  }
      $|}
      $|extern "C" __global__ void \{combine_symbol}(const int *counts,const int *selected_indices,const __nv_bfloat16 *weighted,float *output) {
      $|  int rows=counts[3]; if(rows<=0||rows>\{rows}) return;
      $|  unsigned long long cell=(unsigned long long)blockIdx.x*256+threadIdx.x;
      $|  if(cell>=(unsigned long long)rows*\{hidden}) return;
      $|  unsigned long long row=cell/\{hidden}; int column=cell%\{hidden};
      $|  float sum=0.0f; int previous=-1;
      $|  for(int n=0;n<\{selected};++n) {
      $|    int next=\{experts},slot=-1;
      $|    for(int j=0;j<\{selected};++j) {int expert=selected_indices[row*\{selected}+j];if(expert>previous&&expert<next) {next=expert;slot=j;}}
      $|    if(slot<0) break; sum=\{combine}; previous=next;
      $|  }
      $|  output[cell]=sum;
      $|}
    )
  {
    plan,
    source: @utf8.encode(text),
    symbols: ReadOnlyArray::from_array([
      group_symbol, gate_symbol, down_symbol, combine_symbol,
    ]),
    launches: ReadOnlyArray::from_array([
      (1, 128),
      (jobs * n_gate, 128),
      (jobs * n_down, 128),
      (((rows.to_int64() * hidden.to_int64() + 255L) / 256L).to_int(), 256),
    ]),
  }
}
