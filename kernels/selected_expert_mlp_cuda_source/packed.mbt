///|
pub struct PackedExpertMlpCudaSource {
  priv plan : @precision_ir.ExpertMlpPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
  priv launches : ReadOnlyArray[(Int, Int)]
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
) -> PackedExpertMlpCudaSource raise {
  if symbol.length() > 60 || !valid_selected_expert_symbol(symbol) {
    raise SelectedExpertMlpCudaSourceError::InvalidFunctionSymbol
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
      "gate=__float2bfloat16_rn(fminf(__bfloat162float(gate),\{literal})); up=__float2bfloat16_rn(fmaxf(-\{literal},fminf(__bfloat162float(up),\{literal})));"
    }
  }
  let gate_up = symbol + "_gate_up"
  let down = symbol + "_down"
  let combine = symbol + "_combine"
  let text = loads.join("\n") +
    (
      $|
      $|__device__ __forceinline__ float \{symbol}_reduce(float v) {
      $|  for (int delta=16;delta>0;delta>>=1) v=__fadd_rn(v,__shfl_down_sync(0xffffffffU,v,delta));
      $|  return v;
      $|}
      $|extern "C" __global__ void \{gate_up}(const int *counts, const __nv_bfloat16 *input, const int *selected_indices, const int *expert_map, const unsigned char *bank, __nv_bfloat16 *activation) {
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
      $|    const float x=__bfloat162float(input[row*\{hidden}ULL+column]);
      $|    gate_sum=__fadd_rn(gate_sum,__fmul_rn(x,__bfloat162float(\{symbol}_w0(w0,s0,g0,out,column))));
      $|    up_sum=__fadd_rn(up_sum,__fmul_rn(x,__bfloat162float(\{symbol}_w1(w1,s1,g1,out,column))));
      $|  }
      $|  gate_sum=\{symbol}_reduce(gate_sum); up_sum=\{symbol}_reduce(up_sum);
      $|  if(lane==0) {
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(gate_sum),up=__float2bfloat16_rn(up_sum); \{clamp}
      $|    const float g=__bfloat162float(gate);
      $|    const __nv_bfloat16 activated=__float2bfloat16_rn(__fdiv_rn(g,__fadd_rn(1.0f,expf(-g))));
      $|    activation[cell]=__float2bfloat16_rn(__fmul_rn(__bfloat162float(activated),__bfloat162float(up)));
      $|  }
      $|}
      $|extern "C" __global__ void \{down}(const int *counts, const __nv_bfloat16 *activation, const int *selected_indices, const float *selected_weights, const int *expert_map, const unsigned char *bank, __nv_bfloat16 *weighted) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{maximum_rows}) return;
      $|  const unsigned long long cell=((unsigned long long)blockIdx.x*blockDim.x+threadIdx.x)/32ULL;
      $|  const int lane=threadIdx.x&31; if(cell>=(unsigned long long)live_rows*\{selected}ULL*\{hidden}ULL) return;
      $|  const unsigned long long job=cell/\{hidden}ULL; const int out=(int)(cell%\{hidden}ULL);
      $|  const int expert=selected_indices[job]; const int local=(expert>=0&&expert<\{expert_count})?expert_map[expert]:-1;
      $|  if(local<0||local>=\{plan.local_experts().length()}) { if(lane==0) weighted[cell]=__float2bfloat16_rn(0.0f); return; }
      $|  \{addresses[2]}
      $|  float sum=0.0f; for(int column=lane;column<\{intermediate};column+=32) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(activation[job*\{intermediate}ULL+column]),__bfloat162float(\{symbol}_w2(w2,s2,g2,out,column))));
      $|  sum=\{symbol}_reduce(sum); if(lane==0) weighted[cell]=__float2bfloat16_rn(__fmul_rn(__bfloat162float(__float2bfloat16_rn(sum)),selected_weights[job]));
      $|}
      $|extern "C" __global__ void \{combine}(const int *counts, const int *selected_indices, const __nv_bfloat16 *weighted, float *output) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{maximum_rows}) return;
      $|  const unsigned long long cell=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
      $|  if(cell>=(unsigned long long)live_rows*\{hidden}ULL) return;
      $|  const unsigned long long row=cell/\{hidden}ULL; const int column=(int)(cell%\{hidden}ULL);
      $|  float sum=0.0f; int previous=-1;
      $|  for(int n=0;n<\{selected};++n) {
      $|    int next=\{expert_count},slot=-1;
      $|    for(int j=0;j<\{selected};++j) { const int expert=selected_indices[row*\{selected}ULL+j]; if(expert>previous&&expert<next) {next=expert;slot=j;} }
      $|    if(slot<0) break; sum=__fadd_rn(sum,__bfloat162float(weighted[(row*\{selected}ULL+slot)*\{hidden}ULL+column])); previous=next;
      $|  }
      $|  output[cell]=sum;
      $|}
    )
  {
    plan,
    source: @utf8.encode(text),
    symbols: ReadOnlyArray::from_array([gate_up, down, combine]),
    launches: ReadOnlyArray::from_array([
      (grids[0].to_int(), 128),
      (grids[1].to_int(), 128),
      (grids[2].to_int(), 256),
    ]),
  }
}
