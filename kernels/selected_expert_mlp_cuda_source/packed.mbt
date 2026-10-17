///|
pub struct PackedExpertMlpCudaSource {
  priv plan : @precision_ir.ExpertMlpPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
  priv launches : ReadOnlyArray[(Int, Int)]
  priv packed_rows : @matrix.PackedExpertRows?
  priv bank_preparation : String?
}

///|
/// Optional startup byte-permutation kernel, not part of the token graph.
pub fn PackedExpertMlpCudaSource::bank_preparation(self : Self) -> String? {
  self.bank_preparation
}

///|
pub fn PackedExpertMlpCudaSource::has_packed_rows(self : Self) -> Bool {
  self.packed_rows is Some(_)
}

///|
pub fn PackedExpertMlpCudaSource::plan(
  self : Self,
) -> @precision_ir.ExpertMlpPrecision {
  self.plan
}

///|
pub fn PackedExpertMlpCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn PackedExpertMlpCudaSource::symbols(self : Self) -> ReadOnlyArray[String] {
  self.symbols
}

///|
pub fn PackedExpertMlpCudaSource::launches(
  self : Self,
) -> ReadOnlyArray[(Int, Int)] {
  self.launches
}

///|
/// AOT terminal lowering of the shared precision/expert plan. Warp width and
/// reduction instructions belong only here. The plan's compact bank layout is
/// also consumed by checkpoint upload; formats never switch in the token loop.
pub fn packed_expert_mlp_source(
  plan : @precision_ir.ExpertMlpPrecision,
  symbol~ : String,
  grouping? : @matrix.ExpertGroupingPlan,
) -> PackedExpertMlpCudaSource raise {
  if symbol.length() > 60 || !valid_selected_expert_symbol(symbol) {
    raise SelectedExpertMlpCudaSourceError::InvalidFunctionSymbol
  }
  if plan.scaled_dot_law() is BlockScaledMatrixFloat32(..) {
    return grouped_fp4_source(plan, symbol~, grouping?)
  }
  guard grouping is None ||
    (
      grouping.unwrap().column_tile() == 32 &&
      grouping.unwrap().transport() == IndexedRows
    ) else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  if plan.scaled_dot_law() != DecodeOperandsFloat32 {
    guard grouping is None else {
      raise @precision_ir.PrecisionPlanError::NoImplementation
    }
    return block_scaled_expert_source(plan, symbol~)
  }
  if plan.dot_law() != ScalarFloat32 {
    return grouped_expert_source(plan, symbol~, grouping?)
  }
  guard grouping is None else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let (maximum_rows, hidden, intermediate, expert_count, selected) = plan.geometry()
  let jobs = maximum_rows.to_int64() * selected.to_int64()
  let grids = [
    (jobs * intermediate.to_int64() + 3L) / 4L,
    (jobs * hidden.to_int64() + 3L) / 4L,
    (maximum_rows.to_int64() * hidden.to_int64() + 255L) / 256L,
  ]
  if grids.any(grid => grid <= 0L || grid > 2147483647L) {
    raise SelectedExpertMlpCudaSourceError::DimensionOverflow
  }
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
  let quantizers : Array[String] = []
  let quantizer_symbols : Array[String] = []
  let quantizer_launches : Array[(Int, Int)] = []
  let mut input_type = "const __nv_bfloat16 *"
  let mut activation_type = "const __nv_bfloat16 *"
  let mut input_load = "__bfloat162float(input[row*\{hidden}ULL+column])"
  let mut activation_load = "__bfloat162float(activation[job*\{intermediate}ULL+column])"
  if plan.activation_quantization() is Some((input, product)) {
    for i, stage in [input, product] {
      let name = "\{symbol}_quantize_\{i}"
      let quantizer = @luna_cuda_precision.quantize_bf16_rows_source(
        stage,
        row_multiplier=if i == 0 { 1 } else { selected },
        symbol=name,
      )
      quantizers.push(quantizer.source())
      quantizer_symbols.push(name)
      let launch = quantizer.launch()
      quantizer_launches.push(launch.dimensions())
      let operand = stage.operand()
      let consumer = @precision_ir.plan_operand(
        role=Activation,
        representation=operand.representation(),
        placement=DecodeInOperandTile,
        supported_placements=[DecodeInOperandTile],
        accuracy=operand.accuracy(),
        memory_ceiling=plan.total_bytes(),
        tile_elements=32,
      )
      loads.push(
        @luna_cuda_precision.tile_load_source(
          consumer.refine_region(region),
          symbol="\{symbol}_a\{i}",
        ),
      )
      let offset = @precision_ir.packed_buffer_layout(operand.representation())
        .scales_offset()
        .unwrap()
      if i == 0 {
        input_type = "const unsigned char *"
        input_load = "__bfloat162float(\{symbol}_a0(input,input+\{offset}ULL,(const float *)input,row,column))"
      } else {
        activation_type = "const unsigned char *"
        activation_load = "__bfloat162float(\{symbol}_a1(activation,activation+\{offset}ULL,(const float *)activation,job,column))"
      }
    }
  }
  for i, weight in plan.weights() {
    let refined = weight.refine_region(region)
    loads.push(
      @luna_cuda_precision.tile_load_source(refined, symbol="\{symbol}_w\{i}"),
    )
    let storage = @precision_ir.packed_buffer_layout(weight.representation())
    let scales = storage.scales_offset().unwrap_or(0L)
    let global = storage.global_scale_offset().unwrap_or(0L)
    let offset = plan.weight_offsets()[i]
    addresses.push(
      "const unsigned char *w\{i}=bank+(unsigned long long)local*\{plan.expert_stride()}ULL+\{offset}ULL; " +
      "const unsigned char *s\{i}=w\{i}+\{scales}ULL; const float *g\{i}=(const float *)(w\{i}+\{global}ULL);",
    )
  }
  let clamp = match plan.clamp() {
    None => ""
    Some(limit) => {
      let literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(limit)
      let gate = @luna_cuda_precision.scalar_expression(
        plan.expression(ClampGate),
        ["gate", literal],
      )
      let up = @luna_cuda_precision.scalar_expression(
        plan.expression(ClampUp),
        ["up", literal],
      )
      "gate=\{gate}; up=\{up};"
    }
  }
  let gate_up = symbol + "_gate_up"
  let down = symbol + "_down"
  let combine = symbol + "_combine"
  let route_before = plan.routing_placement() == BeforeDownInputRound
  let routing_parameter = if route_before {
    ", const float *selected_weights"
  } else {
    ""
  }
  fn expr(
    operation : @precision_ir.ExpertExpression,
    bindings : ArrayView[String],
  ) -> String raise {
    @luna_cuda_precision.scalar_expression(plan.expression(operation), bindings)
  }
  let product = "float product=\{expr(Product, ["g", "up"])};"
  let apply_route = if route_before {
    "product=\{expr(ApplyRoute, ["product", "selected_weights[job]"])};"
  } else {
    ""
  }
  let partial = plan.down_publication() == ReduceFloat32BeforeRound
  let weighted_type = if partial { "float" } else { "__nv_bfloat16" }
  let zero = if partial { "0.0f" } else { "__float2bfloat16_rn(0.0f)" }
  let combine_args = if partial {
    ", const float *selected_weights"
  } else {
    ""
  }
  let down_output = if partial {
    "sum"
  } else {
    expr(DownPublish, ["sum", "selected_weights[job]"])
  }
  let gate_update = expr(DotUpdate, [
    "gate_sum",
    "x",
    "__bfloat162float(\{symbol}_w0(w0,s0,g0,out,column))",
  ])
  let up_update = expr(DotUpdate, [
    "up_sum",
    "x",
    "__bfloat162float(\{symbol}_w1(w1,s1,g1,out,column))",
  ])
  let down_update = expr(DotUpdate, [
    "sum",
    activation_load,
    "__bfloat162float(\{symbol}_w2(w2,s2,g2,out,column))",
  ])
  let value = "weighted[(row*\{selected}ULL+slot)*\{hidden}ULL+column]"
  let combine_update = expr(CombineUpdate, [
    "sum",
    if partial {
      expr(DownPublish, [value, "selected_weights[row*\{selected}ULL+slot]"])
    } else {
      value
    },
  ])
  let text = loads.join("\n") +
    (
      $|
      $|__device__ __forceinline__ float \{symbol}_reduce(float v) {
      $|  for (int delta=16;delta>0;delta>>=1) v=__fadd_rn(v,__shfl_down_sync(0xffffffffU,v,delta));
      $|  return v;
      $|}
      $|extern "C" __global__ void \{gate_up}(const int *counts, \{input_type}input, const int *selected_indices, const int *expert_map, const unsigned char *bank, __nv_bfloat16 *activation\{routing_parameter}) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{maximum_rows}) return;
      $|  const unsigned long long cell=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)/32ULL;
      $|  const int lane=threadIdx.x&31; if(cell>=(unsigned long long)live_rows*\{selected}ULL*\{intermediate}ULL) return;
      $|  const unsigned long long job=cell/\{intermediate}ULL; const int out=(int)(cell%\{intermediate}ULL); const unsigned long long row=job/\{selected}ULL;
      $|  const int expert=selected_indices[job]; const int local=(expert>=0&&expert<\{expert_count})?expert_map[expert]:-1;
      $|  if(local<0||local>=\{plan.local_experts().length()}) { if(lane==0) activation[cell]=__float2bfloat16_rn(0.0f); return; }
      $|  \{addresses[0]}
      $|  \{addresses[1]}
      $|  float gate_sum=0.0f,up_sum=0.0f;
      $|  for(int column=lane;column<\{hidden};column+=32) {
      $|    const float x=\{input_load};
      $|    gate_sum=\{gate_update};
      $|    up_sum=\{up_update};
      $|  }
      $|  gate_sum=\{symbol}_reduce(gate_sum); up_sum=\{symbol}_reduce(up_sum);
      $|  if(lane==0) {
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(gate_sum),up=__float2bfloat16_rn(up_sum); \{clamp}
      $|    const float g=__bfloat162float(gate);
      $|    \{product}
      $|    \{apply_route}
      $|    activation[cell]=__float2bfloat16_rn(product);
      $|  }
      $|}
      $|extern "C" __global__ void \{down}(const int *counts, \{activation_type}activation, const int *selected_indices, const float *selected_weights, const int *expert_map, const unsigned char *bank, \{weighted_type} *weighted) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{maximum_rows}) return;
      $|  const unsigned long long cell=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)/32ULL;
      $|  const int lane=threadIdx.x&31; if(cell>=(unsigned long long)live_rows*\{selected}ULL*\{hidden}ULL) return;
      $|  const unsigned long long job=cell/\{hidden}ULL; const int out=(int)(cell%\{hidden}ULL);
      $|  const int expert=selected_indices[job]; const int local=(expert>=0&&expert<\{expert_count})?expert_map[expert]:-1;
      $|  if(local<0||local>=\{plan.local_experts().length()}) { if(lane==0) weighted[cell]=\{zero}; return; }
      $|  \{addresses[2]}
      $|  float sum=0.0f; for(int column=lane;column<\{intermediate};column+=32) sum=\{down_update};
      $|  sum=\{symbol}_reduce(sum); if(lane==0) weighted[cell]=\{down_output};
      $|}
      $|extern "C" __global__ void \{combine}(const int *counts, const int *selected_indices, const \{weighted_type} *weighted, float *output\{combine_args}) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{maximum_rows}) return;
      $|  const unsigned long long cell=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
      $|  if(cell>=(unsigned long long)live_rows*\{hidden}ULL) return;
      $|  const unsigned long long row=cell/\{hidden}ULL; const int column=(int)(cell%\{hidden}ULL);
      $|  float sum=0.0f; int previous=-1;
      $|  for(int n=0;n<\{selected};++n) {
      $|    int next=\{expert_count},slot=-1;
      $|    for(int j=0;j<\{selected};++j) { const int expert=selected_indices[row*\{selected}ULL+j]; if(expert>previous&&expert<next) {next=expert;slot=j;} }
      $|    if(slot<0) break; sum=\{combine_update}; previous=next;
      $|  }
      $|  output[cell]=sum;
      $|}
    )
  let symbols = if quantizers.is_empty() {
    [gate_up, down, combine]
  } else {
    [quantizer_symbols[0], gate_up, quantizer_symbols[1], down, combine]
  }
  let launches = if quantizers.is_empty() {
    [
      (grids[0].to_int(), 128),
      (grids[1].to_int(), 128),
      (grids[2].to_int(), 256),
    ]
  } else {
    [
      quantizer_launches[0],
      (grids[0].to_int(), 128),
      quantizer_launches[1],
      (grids[1].to_int(), 128),
      (grids[2].to_int(), 256),
    ]
  }
  {
    plan,
    packed_rows: None,
    bank_preparation: None,
    source: @utf8.encode(text + "\n" + quantizers.join("\n")),
    symbols: ReadOnlyArray::from_array(symbols),
    launches: ReadOnlyArray::from_array(launches),
  }
}
