///|
/// Native block-scaled matrix lowering. Packing is an explicit preceding
/// effect; there is no per-weight BF16 decode inside the matrix loop.
fn grouped_fp4_source(
  plan : @precision_ir.ExpertMlpPrecision,
  symbol~ : String,
) -> PackedExpertMlpCudaSource raise {
  guard plan.scaled_dot_law() is BlockScaledMatrixFloat32(block_columns=16) &&
    plan.activation_quantization() is Some((qx, qa)) else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let (rows, hidden, intermediate, experts, selected) = plan.geometry()
  let grouping = @matrix.ExpertGroupingPlan::new(
    maximum_rows=rows,
    selected_experts=selected,
  )
  let jobs = grouping.maximum_groups()
  let metadata = (plan.workspace_bytes().0 + 3L) / 4L * 4L
  let prefix = @luna_cuda_precision.e2m1_scaled_matrix_helper(symbol~)
  let quantizers = [qx, qa].mapi((i, q) => {
    @luna_cuda_precision.quantize_fp4_bf16_rows_source(
      q,
      row_multiplier=if i == 0 { 1 } else { selected },
      symbol="\{symbol}_quantize_\{i}",
    )
  })
  let layouts = plan
    .weights()
    .map(w => @precision_ir.packed_buffer_layout(w.representation()))
  let operands = [qx, qa].map(q => {
    @precision_ir.packed_buffer_layout(q.operand().representation())
  })
  let bodies = []
  for stage in 0..<3 {
    let k = if stage < 2 { hidden } else { intermediate }
    let n = if stage < 2 { intermediate } else { hidden }
    let x = operands[if stage < 2 { 0 } else { 1 }]
    let w = layouts[stage]
    // One warp owns a 16x8 output tile. The same logical grouping is used by
    // both projections; inactive rows are zero operands, not live requests.
    bodies.push(
      (
        $|float d\{stage}[4]={0,0,0,0};
        $|const unsigned char *w\{stage}=bank+(unsigned long long)(local<0?0:local)*\{plan.expert_stride()}ULL+\{plan.weight_offsets()[stage]}ULL;
        $|\{fp4_fragment_pipeline(stage,k,n,selected,x.scales_offset().unwrap(),w.scales_offset().unwrap(),symbol)}
        $|float global\{stage}=__fmul_rn(*(const float *)(input+\{x.global_scale_offset().unwrap()}ULL),*(const float *)(w\{stage}+\{w.global_scale_offset().unwrap()}ULL));
        $|for(int i=0;i<4;++i) d\{stage}[i]=__fmul_rn(d\{stage}[i],global\{stage});
      ),
    )
  }
  fn expr(
    op : @precision_ir.ExpertExpression,
    inputs : ArrayView[String],
  ) -> String raise {
    @luna_cuda_precision.scalar_expression(plan.expression(op), inputs)
  }
  let before = plan.routing_placement() == BeforeDownInputRound
  let partial = plan.down_publication() == ReduceFloat32BeforeRound
  let wt = if partial { "float" } else { "__nv_bfloat16" }
  let clamp = match plan.clamp() {
    None => ""
    Some(limit) => {
      let literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(limit)
      "gate=\{expr(ClampGate,["gate",literal])};up=\{expr(ClampUp,["up",literal])};"
    }
  }
  let route = if before {
    "product=" + expr(ApplyRoute, ["product", "selected_weights[job]"]) + ";"
  } else {
    ""
  }
  let down = if partial {
    "d2[i]"
  } else {
    expr(DownPublish, ["d2[i]", "selected_weights[job]"])
  }
  let value = "weighted[(row*\{selected}ULL+slot)*\{hidden}+col]"
  let addition = expr(CombineUpdate, [
    "sum",
    if partial {
      expr(DownPublish, [value, "selected_weights[row*\{selected}ULL+slot]"])
    } else {
      value
    },
  ])
  let ng = (intermediate + 31) / 32
  let nd = (hidden + 31) / 32
  guard jobs.to_int64() * ng.to_int64() <= 2147483647L &&
    jobs.to_int64() * nd.to_int64() <= 2147483647L else {
    raise @precision_ir.PrecisionPlanError::ArithmeticOverflow
  }
  let text = quantizers.join("\n") +
    prefix +
    grouping_kernel(symbol + "_group", rows, selected, metadata) +
    (
      $|extern "C" __global__ void \{symbol}_gate_up(const int *counts,const unsigned char *input,const int *indices,const int *expert_map,const unsigned char *bank,__nv_bfloat16 *activation\{if before {",const float *selected_weights"} else {""}}) {
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);int group=blockIdx.x/\{ng};if(group>=meta[0]) return;
      $|  int lane=threadIdx.x&31,g=lane/4,t=lane%4,column=(blockIdx.x%\{ng})*32+(threadIdx.x/32)*8;
      $|  int expert=meta[1+group],local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{bodies[0]}
      $|  \{bodies[1]}
      $|  for(int i=0;i<4;++i) {int job=job_ids[g+(i/2)*8],out=column+t*2+i%2;if(job<0||out>=\{intermediate}) continue;
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(d0[i]),up=__float2bfloat16_rn(d1[i]);\{clamp}
      $|    float g=__bfloat162float(gate),product=\{expr(Product,["g","up"])};\{route}
      $|    activation[(unsigned long long)job*\{intermediate}+out]=local<0?__float2bfloat16_rn(0.0f):__float2bfloat16_rn(product);
      $|  }
      $|}
      $|extern "C" __global__ void \{symbol}_down(const int *counts,const unsigned char *input,const int *indices,const float *selected_weights,const int *expert_map,const unsigned char *bank,\{wt} *weighted,const __nv_bfloat16 *activation) {
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);int group=blockIdx.x/\{nd};if(group>=meta[0]) return;
      $|  int lane=threadIdx.x&31,g=lane/4,t=lane%4,column=(blockIdx.x%\{nd})*32+(threadIdx.x/32)*8;
      $|  int expert=meta[1+group],local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{bodies[2]}
      $|  for(int i=0;i<4;++i) {int job=job_ids[g+(i/2)*8],out=column+t*2+i%2;if(job>=0&&out<\{hidden}) weighted[(unsigned long long)job*\{hidden}+out]=\{down};}
      $|}
      $|extern "C" __global__ void \{symbol}_combine(const int *counts,const int *indices,const \{wt} *weighted,float *output\{if partial {",const float *selected_weights"} else {""}}) {
      $|  int rows=counts[3];if(rows<=0||rows>\{rows}) return;unsigned long long cell=(unsigned long long)blockIdx.x*256+threadIdx.x;if(cell>=(unsigned long long)rows*\{hidden}) return;
      $|  unsigned long long row=cell/\{hidden};int col=cell%\{hidden};float sum=0;int previous=-1;
      $|  for(int n=0;n<\{selected};++n) {int next=\{experts},slot=-1;for(int j=0;j<\{selected};++j) {int expert=indices[row*\{selected}+j];if(expert>previous&&expert<next) {next=expert;slot=j;}}if(slot<0) break;sum=\{addition};previous=next;}output[cell]=sum;
      $|}
    )
  {
    plan,
    source: @utf8.encode(text),
    symbols: ReadOnlyArray::from_array([
      symbol + "_group",
      symbol + "_quantize_0_global",
      symbol + "_quantize_0_pack",
      symbol + "_gate_up",
      symbol + "_quantize_1_global",
      symbol + "_quantize_1_pack",
      symbol + "_down",
      symbol + "_combine",
    ]),
    launches: ReadOnlyArray::from_array([
      (1, 128),
      (1, 256),
      (((rows.to_int64() * hidden.to_int64() + 127L) / 128L).to_int(), 128),
      (jobs * ng, 128),
      (1, 256),
      (
        ((jobs.to_int64() * intermediate.to_int64() + 127L) / 128L).to_int(),
        128,
      ),
      (jobs * nd, 128),
      (((rows.to_int64() * hidden.to_int64() + 255L) / 256L).to_int(), 256),
    ]),
  }
}
