///|
/// Native block-scaled matrix lowering. Packing is an explicit preceding
/// effect; there is no per-weight BF16 decode inside the matrix loop.
fn grouped_fp4_source(
  plan : @precision_ir.ExpertMlpPrecision,
  symbol~ : String,
  grouping? : @matrix.ExpertGroupingPlan,
) -> PackedExpertMlpCudaSource raise {
  guard plan.scaled_dot_law() is BlockScaledMatrixFloat32(block_columns=16) &&
    plan.activation_quantization() is Some((qx, qa)) else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let (rows, hidden, intermediate, experts, selected) = plan.geometry()
  let grouping = grouping.unwrap_or(
    @matrix.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
    ),
  )
  guard grouping.maximum_groups().to_int64() ==
    rows.to_int64() * selected.to_int64() else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let jobs = grouping.maximum_groups()
  let tile_columns = grouping.column_tile()
  guard tile_columns is (32 | 64 | 128) else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let accumulators = tile_columns / 8
  // Fold the publication coordinates over a single-fragment index domain.
  // Keep its established loop policy; explicit unrolling is for wider owners.
  let unroll = if accumulators == 4 { "" } else { "#pragma unroll" }
  let row_half = if accumulators == 4 { "i/2" } else { "i%4/2" }
  let column_offset = if accumulators == 4 { "" } else { "(i/4)*32+" }
  let metadata = (plan.workspace_bytes().0 + 3L) / 4L * 4L
  let packed_rows = if grouping.transport() is (IndexedRows | IndexedBuffered) {
    None
  } else {
    Some(
      @matrix.PackedExpertRows::new(
        grouping,
        after_bytes=metadata + grouping.storage_bytes(),
        payload_bytes=(hidden / 2, intermediate / 2),
        scale_bytes=(hidden / 16, intermediate / 16),
      ),
    )
  }
  let pipeline = if grouping.transport() == PackedPipeline {
    Some(
      @physical_tile_ir.ProducerConsumerPipeline::new(
        stages=2,
        producers=128,
        consumers=128,
      ).unwrap(),
    )
  } else {
    None
  }
  let matrix_threads = if pipeline is Some(_) { 256 } else { 128 }
  let shared_ring = if grouping.transport() == IndexedBuffered &&
    !grouping.shared_input_staging() {
    let layout = @physical_tile_ir.OperandLayout::new(
      columns=128,
      vector=16,
      rotations=8,
      row_group=1,
    ).unwrap()
    let storage = @physical_tile_ir.OperandStorage::new(
      layout,
      [16, tile_columns],
      owners=128,
      stages=2,
    ).unwrap()
    let bytes = (storage.slot_elements() + 256 + tile_columns * 16).to_int64() *
      2L
    let lifetime = @physical_tile_ir.SequentialScratch::new(
      producer_bytes=bytes,
      consumer_bytes=bytes,
      overlapping=false,
    ).unwrap()
    "__shared__ __align__(16) unsigned char gate_up_operands[\{lifetime.allocation_bytes()}];"
  } else {
    ""
  }
  let consumer = if pipeline is Some(_) { "threadIdx.x>=128" } else { "true" }
  let warp_owner = if pipeline is Some(_) {
    "(threadIdx.x%128)"
  } else {
    "threadIdx.x"
  }
  let packed_pointer = match packed_rows {
    None => ""
    Some(p) =>
      "const unsigned char *packed=(const unsigned char *)activation+\{p.base()}ULL;"
  }
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
    // Each warp owns one or more independent 16x8 fragments. Both projections
    // share logical grouping; inactive rows are zero operands, not requests.
    bodies.push(
      (
        $|float d\{stage}[\{accumulators}]={0};
        $|const unsigned char *w\{stage}=bank+(unsigned long long)(local<0?0:local)*\{plan.expert_stride()}ULL+\{plan.weight_offsets()[stage]}ULL;
        $|\{fp4_fragment_pipeline(stage,k,n,selected,packed_rows.map(p => p.plane(stage==2).2).unwrap_or(x.scales_offset().unwrap()),w.scales_offset().unwrap(),symbol,work_tile="work_tile",tile_columns~,packed_input=packed_rows is Some(_),pipeline?,buffered=grouping.transport()==IndexedBuffered,borrowed_ring?=if stage<2&&!shared_ring.is_empty() {Some("gate_up_operands")} else {None},fragment_slots=grouping.fragment_slots())}
        $|float global\{stage}=__fmul_rn(*(const float *)(input+\{x.global_scale_offset().unwrap()}ULL),*(const float *)(w\{stage}+\{w.global_scale_offset().unwrap()}ULL));
        $|\{unroll}
        $|for(int i=0;i<\{accumulators};++i) d\{stage}[i]=__fmul_rn(d\{stage}[i],global\{stage});
      ),
    )
  }
  // Reuse the logical input owner across two independent ordered reductions.
  // This changes physical staging only, not either projection's arithmetic.
  let gate_up = if grouping.shared_input_staging() {
    let x = operands[0]
    let declarations = [0, 1]
      .map(stage => {
        (
          $|float d\{stage}[\{accumulators}]={0};
          $|const unsigned char *w\{stage}=bank+(unsigned long long)(local<0?0:local)*\{plan.expert_stride()}ULL+\{plan.weight_offsets()[stage]}ULL;
        )
      })
      .join("\n")
    let publication = [0, 1]
      .map(stage => {
        (
          $|float global\{stage}=__fmul_rn(*(const float *)(input+\{x.global_scale_offset().unwrap()}ULL),*(const float *)(w\{stage}+\{layouts[stage].global_scale_offset().unwrap()}ULL));
          $|\{unroll}
          $|for(int i=0;i<\{accumulators};++i) d\{stage}[i]=__fmul_rn(d\{stage}[i],global\{stage});
        )
      })
      .join("\n")
    declarations +
    "\n" +
    fp4_fragment_pipeline(
      0,
      hidden,
      intermediate,
      selected,
      packed_rows
      .map(p => p.plane(false).2)
      .unwrap_or(x.scales_offset().unwrap()),
      layouts[0].scales_offset().unwrap(),
      symbol,
      work_tile="work_tile",
      paired_weight_scales=layouts[1].scales_offset().unwrap(),
      tile_columns~,
      packed_input=packed_rows is Some(_),
      pipeline?,
      buffered=grouping.transport() == IndexedBuffered,
      fragment_slots=grouping.fragment_slots(),
    ) +
    "\n" +
    publication
  } else {
    bodies[0] + "\n" + bodies[1]
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
  let ng = (intermediate + tile_columns - 1) / tile_columns
  let nd = (hidden + tile_columns - 1) / tile_columns
  let gate_tiles = @matrix.IndirectMatrixTraversal::new(
    capacity_groups=jobs,
    column_tiles=ng,
  )
  let down_tiles = @matrix.IndirectMatrixTraversal::new(
    capacity_groups=jobs,
    column_tiles=nd,
  )
  guard jobs.to_int64() * ng.to_int64() <= 2147483647L &&
    jobs.to_int64() * nd.to_int64() <= 2147483647L else {
    raise @precision_ir.PrecisionPlanError::ArithmeticOverflow
  }
  let text = quantizers.join("\n") +
    prefix +
    grouping_kernel(symbol + "_group", rows, selected, metadata) +
    (match packed_rows {
      None => ""
      Some(p) =>
        [0, 1]
        .map(i => {
          fp4_pack_rows_source(
            p,
            i == 1,
            jobs,
            selected,
            metadata,
            operands[i].scales_offset().unwrap(),
            symbol + "_pack_rows_" + i.to_string(),
          )
        })
        .join("\n")
    }) +
    (
      $|extern "C" __global__ void \{symbol}_gate_up(const int *counts,const unsigned char *input,const int *indices,const int *expert_map,const unsigned char *bank,__nv_bfloat16 *activation\{if before {",const float *selected_weights"} else {""}}) {
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);
      $|  \{packed_pointer}
      $|  \{shared_ring}
      $|  for(int work_tile=blockIdx.x;work_tile<meta[0]*\{ng};work_tile+=gridDim.x) { int group=work_tile/\{ng};
      $|  int lane=threadIdx.x&31,g=lane/4,t=lane%4,column=(work_tile%\{ng})*\{tile_columns}+(\{warp_owner}/32)*8;
      $|  int expert=meta[1+group],local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{gate_up}
      $|  if(\{consumer}) {
      $|  \{unroll}
      $|  for(int i=0;i<\{accumulators};++i) {int job=job_ids[g+(\{row_half})*8],out=column+\{column_offset}t*2+i%2;if(job<0||out>=\{intermediate}) continue;
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(d0[i]),up=__float2bfloat16_rn(d1[i]);\{clamp}
      $|    float g=__bfloat162float(gate),product=\{expr(Product,["g","up"])};\{route}
      $|    activation[(unsigned long long)job*\{intermediate}+out]=local<0?__float2bfloat16_rn(0.0f):__float2bfloat16_rn(product);
      $|  }
      $|  }
      $|  }
      $|}
      $|extern "C" __global__ void \{symbol}_down(const int *counts,const unsigned char *input,const int *indices,const float *selected_weights,const int *expert_map,const unsigned char *bank,\{wt} *weighted,const __nv_bfloat16 *activation) {
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);
      $|  \{packed_pointer}
      $|  for(int work_tile=blockIdx.x;work_tile<meta[0]*\{nd};work_tile+=gridDim.x) { int group=work_tile/\{nd};
      $|  int lane=threadIdx.x&31,g=lane/4,t=lane%4,column=(work_tile%\{nd})*\{tile_columns}+(\{warp_owner}/32)*8;
      $|  int expert=meta[1+group],local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{bodies[2]}
      $|  if(\{consumer}) {
      $|  \{unroll}
      $|  for(int i=0;i<\{accumulators};++i) {int job=job_ids[g+(\{row_half})*8],out=column+\{column_offset}t*2+i%2;if(job>=0&&out<\{hidden}) weighted[(unsigned long long)job*\{hidden}+out]=\{down};}
      $|  }
      $|  }
      $|}
      $|extern "C" __global__ void \{symbol}_combine(const int *counts,const int *indices,const \{wt} *weighted,float *output\{if partial {",const float *selected_weights"} else {""}}) {
      $|  int rows=counts[3];if(rows<=0||rows>\{rows}) return;unsigned long long cell=(unsigned long long)blockIdx.x*256+threadIdx.x;if(cell>=(unsigned long long)rows*\{hidden}) return;
      $|  unsigned long long row=cell/\{hidden};int col=cell%\{hidden};float sum=0;int previous=-1;
      $|  for(int n=0;n<\{selected};++n) {int next=\{experts},slot=-1;for(int j=0;j<\{selected};++j) {int expert=indices[row*\{selected}+j];if(expert>previous&&expert<next) {next=expert;slot=j;}}if(slot<0) break;sum=\{addition};previous=next;}output[cell]=sum;
      $|}
    )
  let symbols = [
    symbol + "_group",
    symbol + "_quantize_0_global",
    symbol + "_quantize_0_pack",
  ]
  let launches = [
    (1, 128),
    (1, 256),
    (((rows.to_int64() * hidden.to_int64() + 127L) / 128L).to_int(), 128),
  ]
  if packed_rows is Some(_) {
    symbols.push(symbol + "_pack_rows_0")
    launches.push((jobs, 128))
  }
  symbols.append([
    symbol + "_gate_up",
    symbol + "_quantize_1_global",
    symbol + "_quantize_1_pack",
  ])
  launches.append([
    (gate_tiles.workers(), matrix_threads),
    (1, 256),
    (((jobs.to_int64() * intermediate.to_int64() + 127L) / 128L).to_int(), 128),
  ])
  if packed_rows is Some(_) {
    symbols.push(symbol + "_pack_rows_1")
    launches.push((jobs, 128))
  }
  symbols.append([symbol + "_down", symbol + "_combine"])
  launches.append([
    (down_tiles.workers(), matrix_threads),
    (((rows.to_int64() * hidden.to_int64() + 255L) / 256L).to_int(), 256),
  ])
  {
    plan,
    packed_rows,
    source: @utf8.encode(text),
    symbols: ReadOnlyArray::from_array(symbols),
    launches: ReadOnlyArray::from_array(launches),
  }
}
