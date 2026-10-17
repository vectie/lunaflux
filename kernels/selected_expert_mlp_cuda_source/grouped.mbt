///|
/// The extra region stores a bounded physical grouping, not model activations.
/// It is allocated once with the activation frame and survives gate/down.
pub fn PackedExpertMlpCudaSource::activation_bytes(self : Self) -> Int64 {
  if self.packed_rows is Some(rows) {
    return rows.total_bytes()
  }
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
  grouping? : @matrix.ExpertGroupingPlan,
) -> PackedExpertMlpCudaSource raise {
  guard plan.activation_quantization() is None &&
    plan.scaled_dot_law() == DecodeOperandsFloat32 else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
  let (rows, hidden, intermediate, experts, selected) = plan.geometry()
  let grouping = grouping.unwrap_or(
    @matrix.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
      shared_input_staging=plan.dot_law() == ShapeAwareFloat32,
    ),
  )
  guard grouping.maximum_groups().to_int64() ==
    rows.to_int64() * selected.to_int64() else {
    raise @precision_ir.PrecisionPlanError::NoImplementation
  }
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
  let vectors : Array[Bool] = []
  let addresses : Array[String] = []
  for i, weight in plan.weights() {
    loads.push(
      @luna_cuda_precision.tile_load_source(
        weight.refine_region(region),
        symbol="\{symbol}_w\{i}",
      ),
    )
    let vector = @luna_cuda_precision.tile_load_vector_source(
      weight.refine_region(region),
      symbol="\{symbol}_v\{i}",
    )
    vectors.push(vector is Some(_))
    if vector is Some(code) {
      loads.push(code)
    }
    let storage = @precision_ir.packed_buffer_layout(weight.representation())
    let offset = plan.weight_offsets()[i]
    let scales = storage.scales_offset().unwrap_or(0L)
    let global = storage.global_scale_offset().unwrap_or(0L)
    addresses.push(
      "const unsigned char *w\{i}=bank+(unsigned long long)(local<0?0:local)*\{plan.expert_stride()}ULL+\{offset}ULL; const unsigned char *s\{i}=w\{i}+\{scales}ULL; const float *g\{i}=(const float *)(w\{i}+\{global}ULL);",
    )
  }
  let input = "(job_ids[matrix_row]>=0&&local>=0?input[(unsigned long long)(job_ids[matrix_row]/\{selected})*\{hidden}+matrix_inner]:__float2bfloat16_rn(0.0f))"
  let pair_weights : Array[String] = []
  let pair_vectors : Array[String] = []
  for i in 0..<2 {
    pair_weights.push(
      "(local>=0&&column_base+matrix_column<\{intermediate}?\{symbol}_w\{i}(w\{i},s\{i},g\{i},column_base+matrix_column,matrix_inner):__float2bfloat16_rn(0.0f))",
    )
    pair_vectors.push(
      "(local>=0&&column_base+matrix_column<\{intermediate}?\{symbol}_v\{i}(w\{i},s\{i},g\{i},column_base+matrix_column,matrix_inner):make_uint4(0,0,0,0))",
    )
  }
  let pair_body = if grouping.shared_input_staging() {
    @tiled.paired_matrix_tile_body(
      gate_plan,
      input~,
      weights=(pair_weights[0], pair_weights[1]),
      results=("result0", "result1"),
      vector_guard="((unsigned long long)input&15ULL)==0ULL",
      input_vector?=if hidden % 8 == 0 {
        Some(
          "(job_ids[matrix_row]>=0&&local>=0?*reinterpret_cast<const uint4 *>(input+(unsigned long long)(job_ids[matrix_row]/\{selected})*\{hidden}+matrix_inner):make_uint4(0,0,0,0))",
        )
      } else {
        None
      },
      weight_vectors?=if vectors[0] && vectors[1] {
        Some((pair_vectors[0], pair_vectors[1]))
      } else {
        None
      },
    )
  } else {
    let bodies = []
    for i in 0..<2 {
      bodies.push(
        @tiled.matrix_tile_body(
          gate_plan,
          input~,
          weight=pair_weights[i],
          result="result\{i}",
          vector_guard="((unsigned long long)input&15ULL)==0ULL",
          input_vector?=if hidden % 8 == 0 {
            Some(
              "(job_ids[matrix_row]>=0&&local>=0?*reinterpret_cast<const uint4 *>(input+(unsigned long long)(job_ids[matrix_row]/\{selected})*\{hidden}+matrix_inner):make_uint4(0,0,0,0))",
            )
          } else {
            None
          },
          weight_vector?=if vectors[i] { Some(pair_vectors[i]) } else { None },
        ),
      )
    }
    bodies.join("\n")
  }
  let down_body = @tiled.matrix_tile_body(
    down_plan,
    input="(job_ids[matrix_row]>=0&&local>=0?activation[(unsigned long long)job_ids[matrix_row]*\{intermediate}+matrix_inner]:__float2bfloat16_rn(0.0f))",
    weight="(local>=0&&column_base+matrix_column<\{hidden}?\{symbol}_w2(w2,s2,g2,column_base+matrix_column,matrix_inner):__float2bfloat16_rn(0.0f))",
    result="result",
    vector_guard="((unsigned long long)activation&15ULL)==0ULL",
    input_vector?=if intermediate % 8 == 0 {
      Some(
        "(job_ids[matrix_row]>=0&&local>=0?*reinterpret_cast<const uint4 *>(activation+(unsigned long long)job_ids[matrix_row]*\{intermediate}+matrix_inner):make_uint4(0,0,0,0))",
      )
    } else {
      None
    },
    weight_vector?=if vectors[2] {
      Some(
        "(local>=0&&column_base+matrix_column<\{hidden}?\{symbol}_v2(w2,s2,g2,column_base+matrix_column,matrix_inner):make_uint4(0,0,0,0))",
      )
    } else {
      None
    },
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
  let partial = plan.down_publication() == ReduceFloat32BeforeRound
  let weighted_type = if partial { "float" } else { "__nv_bfloat16" }
  let zero = if partial { "0.0f" } else { "__float2bfloat16_rn(0.0f)" }
  let publish = if partial {
    "result[i]"
  } else {
    expr(DownPublish, ["result[i]", "selected_weights[job]"])
  }
  let combine_args = if partial { ",const float *selected_weights" } else { "" }
  let value = "weighted[(row*\{selected}ULL+slot)*\{hidden}ULL+column]"
  let combine = expr(CombineUpdate, [
    "sum",
    if partial {
      expr(DownPublish, [value, "selected_weights[row*\{selected}ULL+slot]"])
    } else {
      value
    },
  ])
  let group_symbol = symbol + "_group"
  let gate_symbol = symbol + "_gate_up"
  let down_symbol = symbol + "_down"
  let combine_symbol = symbol + "_combine"
  let n_gate = (intermediate + 63) / 64
  let n_down = (hidden + 63) / 64
  guard plan.dot_law()!=ShapeAwareFloat32 ||
    (selected.to_int64()*intermediate.to_int64()<=2147483644L &&
     selected.to_int64()*hidden.to_int64()<=2147483644L) else {
    raise @precision_ir.PrecisionPlanError::ArithmeticOverflow
  }
  let (decode_gate, decode_down) = if plan.dot_law() == ShapeAwareFloat32 {
    shape_expert_decode(
      plan,
      symbol~,
      addresses~,
      product~,
      clamp~,
      route~,
      zero~,
    )
  } else {
    ("", "")
  }
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
      $|  \{decode_gate}
      $|  const int *meta=(const int *)((const unsigned char *)activation+\{metadata}ULL);
      $|  const int group=blockIdx.x/\{n_gate},column_base=(blockIdx.x%\{n_gate})*64;
      $|  if(group>=meta[0]) return;
      $|  const int expert=meta[1+group]; int local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1;
      $|  if(local>=\{plan.local_experts().length()}) local=-1;
      $|  const int *job_ids=meta+1+\{jobs}+group*16;
      $|  \{addresses[0]}
      $|  \{addresses[1]}
      $|  __shared__ __align__(32) float result0[1024],result1[1024];
      $|  \{pair_body}
      $|  for(int i=threadIdx.x;i<1024;i+=128) {
      $|    int job=job_ids[i/64],column=column_base+i%64;
      $|    if(job<0||column>=\{intermediate}) continue;
      $|    __nv_bfloat16 gate=__float2bfloat16_rn(result0[i]),up=__float2bfloat16_rn(result1[i]); \{clamp}
      $|    float g=__bfloat162float(gate),product=\{product}; \{route}
      $|    activation[(unsigned long long)job*\{intermediate}+column]=local<0?__float2bfloat16_rn(0.0f):__float2bfloat16_rn(product);
      $|  }
      $|}
      $|extern "C" __global__ void \{down_symbol}(const int *counts,const __nv_bfloat16 *activation,const int *selected_indices,const float *selected_weights,const int *expert_map,const unsigned char *bank,\{weighted_type} *weighted) {
      $|  \{decode_down}
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
      $|    if(job>=0&&column<\{hidden}) weighted[(unsigned long long)job*\{hidden}+column]=local<0?\{zero}:\{publish};
      $|  }
      $|}
      $|extern "C" __global__ void \{combine_symbol}(const int *counts,const int *selected_indices,const \{weighted_type} *weighted,float *output\{combine_args}) {
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
    packed_rows: None,
    symbols: ReadOnlyArray::from_array([
      group_symbol, gate_symbol, down_symbol, combine_symbol,
    ]),
    launches: ReadOnlyArray::from_array([
      (1, 128),
      (
        if plan.dot_law() == ShapeAwareFloat32 {
          (jobs * n_gate).max((selected * intermediate + 3) / 4)
        } else {
          jobs * n_gate
        },
        128,
      ),
      (
        if plan.dot_law() == ShapeAwareFloat32 {
          (jobs * n_down).max((selected * hidden + 3) / 4)
        } else {
          jobs * n_down
        },
        128,
      ),
      (((rows.to_int64() * hidden.to_int64() + 255L) / 256L).to_int(), 256),
    ]),
  }
}
