///|
/// Private E4M3 terminal lowering. Logical blocks/scaled publication live in
/// precision IR; warp ownership, neutral physical rows and MMA ABI live here.
fn block_scaled_expert_source(
  plan : @precision_ir.ExpertMlpPrecision,
  symbol~ : String,
) -> PackedExpertMlpCudaSource raise {
  guard plan.scaled_dot_law()
    is UnscaledBlockMatrixFloat32(block_columns=128, publication~) &&
    plan.activation_quantization() is Some((input, product)) else {
    raise @luna_cuda_precision.LoweringError::UnsupportedGeometry
  }
  let (maximum_rows, hidden, intermediate, experts, selected) = plan.geometry()
  for weight in plan.weights() {
    let rep = weight.representation()
    guard rep.element() is E4M3 &&
      rep.scale_element() is Ue8M0 &&
      rep.grouping() is Block(rows=128, columns=128) &&
      !rep.has_global_scale() else {
      raise @luna_cuda_precision.LoweringError::UnsupportedGeometry
    }
  }
  for stage in [input, product] {
    let rep = stage.operand().representation()
    guard rep.element() is E4M3 &&
      rep.scale_element() is Ue8M0 &&
      rep.grouping() is Block(rows=1, columns=128) &&
      !rep.has_global_scale() else {
      raise @luna_cuda_precision.LoweringError::UnsupportedGeometry
    }
  }
  let quantizers = [input, product].mapi((i, stage) => {
    @luna_cuda_precision.quantize_bf16_rows_source(
      stage,
      row_multiplier=if i == 0 { 1 } else { selected },
      symbol="\{symbol}_quantize_\{i}",
    )
  })
  let scale_offsets = [input, product].map(stage => {
    @precision_ir.packed_buffer_layout(stage.operand().representation())
    .scales_offset()
    .unwrap()
  })
  let offsets = plan.weight_offsets()
  let helper = @luna_cuda_precision.e4m3_block_matrix_helper(
    symbol~,
    publication~,
  )
  let clamp = match plan.clamp() {
    None => ""
    Some(limit) => {
      let value = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(limit)
      "gate=__float2bfloat16_rn(fminf(__bfloat162float(gate),\{value}));up=__float2bfloat16_rn(fmaxf(-\{value},fminf(__bfloat162float(up),\{value})));"
    }
  }
  let route_before = plan.routing_placement() == BeforeDownInputRound
  let product_expression = match plan.activation_rounding() {
    Bf16Stages =>
      "__fmul_rn(__bfloat162float(__float2bfloat16_rn(__fdiv_rn(g,__fadd_rn(1.0f,expf(-g))))),__bfloat162float(up))"
    F32Product =>
      "__fmul_rn(__fdiv_rn(g,__fadd_rn(1.0f,expf(-g))),__bfloat162float(up))"
  }
  let route_parameter = if route_before {
    ",const float *selected_weights"
  } else {
    ""
  }
  let apply_route = if route_before {
    "v=__fmul_rn(v,selected_weights[job]);"
  } else {
    ""
  }
  let down_output = if route_before {
    "__float2bfloat16_rn(sums[i])"
  } else {
    "__float2bfloat16_rn(__fmul_rn(__bfloat162float(__float2bfloat16_rn(sums[i])),selected_weights[job]))"
  }
  let gate_up = symbol + "_gate_up"
  let down = symbol + "_down"
  let combine = symbol + "_combine"
  let text = helper +
    (
      $|
      $|extern "C" __global__ void \{gate_up}(const int *counts,const unsigned char *input,const int *selected_indices,const int *expert_map,const unsigned char *bank,__nv_bfloat16 *activation\{route_parameter}) {
      $|  const int live=counts[3];if(live<=0||live>\{maximum_rows})return;
      $|  const unsigned long long cell=(unsigned long long)blockIdx.x*4+threadIdx.x/32,job=cell/\{intermediate/8}ULL;
      $|  if(job>=(unsigned long long)live*\{selected}ULL)return;
      $|  const int tile=(cell%\{intermediate/8}ULL)*8,lane=threadIdx.x&31,g=lane/4,t=lane%4;
      $|  const int expert=selected_indices[job],local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;
      $|  if(local<0||local>=\{plan.local_experts().length()}) {if(g==0)for(int i=0;i<2;++i)activation[job*\{intermediate}ULL+tile+t*2+i]=__float2bfloat16_rn(0.0f);return;}
      $|  const unsigned long long row=job/\{selected}ULL;
      $|  const unsigned char *x=input+row*\{hidden}ULL,*xs=input+\{scale_offsets[0]}ULL+row*\{hidden/128}ULL;
      $|  const unsigned char *w0=bank+(unsigned long long)local*\{plan.expert_stride()}ULL+\{offsets[0]}ULL,*w1=bank+(unsigned long long)local*\{plan.expert_stride()}ULL+\{offsets[1]}ULL;
      $|  float gates[2],ups[2];\{symbol}_block_dot<\{hidden}>(x,xs,w0,w0+\{hidden.to_int64()*intermediate.to_int64()}ULL,tile,gates);\{symbol}_block_dot<\{hidden}>(x,xs,w1,w1+\{hidden.to_int64()*intermediate.to_int64()}ULL,tile,ups);
      $|  if(g==0)for(int i=0;i<2;++i) {
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(gates[i]),up=__float2bfloat16_rn(ups[i]);\{clamp}
      $|    const float g=__bfloat162float(gate);float v=\{product_expression};\{apply_route}
      $|    activation[job*\{intermediate}ULL+tile+t*2+i]=__float2bfloat16_rn(v);
      $|  }
      $|}
      $|extern "C" __global__ void \{down}(const int *counts,const unsigned char *activation,const int *selected_indices,const float *selected_weights,const int *expert_map,const unsigned char *bank,__nv_bfloat16 *weighted) {
      $|  const int live=counts[3];if(live<=0||live>\{maximum_rows})return;
      $|  const unsigned long long cell=(unsigned long long)blockIdx.x*4+threadIdx.x/32,job=cell/\{hidden/8}ULL;if(job>=(unsigned long long)live*\{selected}ULL)return;
      $|  const int tile=(cell%\{hidden/8}ULL)*8,lane=threadIdx.x&31,g=lane/4,t=lane%4;
      $|  const int expert=selected_indices[job],local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;
      $|  if(local<0||local>=\{plan.local_experts().length()}) {if(g==0)for(int i=0;i<2;++i)weighted[job*\{hidden}ULL+tile+t*2+i]=__float2bfloat16_rn(0.0f);return;}
      $|  const unsigned char *x=activation+job*\{intermediate}ULL,*xs=activation+\{scale_offsets[1]}ULL+job*\{intermediate/128}ULL;
      $|  const unsigned char *w=bank+(unsigned long long)local*\{plan.expert_stride()}ULL+\{offsets[2]}ULL;
      $|  float sums[2];\{symbol}_block_dot<\{intermediate}>(x,xs,w,w+\{hidden.to_int64()*intermediate.to_int64()}ULL,tile,sums);
      $|  if(g==0)for(int i=0;i<2;++i)weighted[job*\{hidden}ULL+tile+t*2+i]=\{down_output};
      $|}
      $|extern "C" __global__ void \{combine}(const int *counts,const int *selected_indices,const __nv_bfloat16 *weighted,float *output) {
      $|  const int live=counts[3];if(live<=0||live>\{maximum_rows})return;
      $|  const unsigned long long cell=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(cell>=(unsigned long long)live*\{hidden}ULL)return;
      $|  const unsigned long long row=cell/\{hidden}ULL;const int column=cell%\{hidden}ULL;float sum=0.0f;int previous=-1;
      $|  for(int n=0;n<\{selected};++n) {int next=\{experts},slot=-1;for(int j=0;j<\{selected};++j){int e=selected_indices[row*\{selected}ULL+j];if(e>previous&&e<next){next=e;slot=j;}}if(slot<0)break;sum=__fadd_rn(sum,__bfloat162float(weighted[(row*\{selected}ULL+slot)*\{hidden}ULL+column]));previous=next;}
      $|  output[cell]=sum;
      $|}
    ) +
    "\n" +
    quantizers.map(q => q.source()).join("\n")
  let jobs = maximum_rows.to_int64() * selected.to_int64()
  let grids = [
    (jobs * intermediate.to_int64() + 31L) / 32L,
    (jobs * hidden.to_int64() + 31L) / 32L,
    (maximum_rows.to_int64() * hidden.to_int64() + 255L) / 256L,
  ]
  guard grids.all(grid => grid > 0L && grid <= 2147483647L) else {
    raise SelectedExpertMlpCudaSourceError::DimensionOverflow
  }
  {
    plan,
    source: @utf8.encode(text),
    symbols: [
      quantizers[0].symbol(),
      gate_up,
      quantizers[1].symbol(),
      down,
      combine,
    ],
    launches: [
      quantizers[0].launch().dimensions(),
      (grids[0].to_int(), 128),
      quantizers[1].launch().dimensions(),
      (grids[1].to_int(), 128),
      (grids[2].to_int(), 256),
    ],
  }
}
