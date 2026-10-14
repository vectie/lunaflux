///|
/// Live single-row experts use the same projection strategy as dense GEMV.
/// Group formation remains bounded metadata and multirow execution keeps the
/// existing matrix representation. No model names or learned measurements.
fn shape_expert_decode(
  plan : @precision_ir.ExpertMlpPrecision,
  symbol~ : String,
  addresses~ : Array[String],
  product~ : String,
  clamp~ : String,
  route~ : String,
  zero~ : String,
) -> (String, String) raise {
  let (rows, hidden, intermediate, experts, selected) = plan.geometry()
  fn prepare(width : Int) -> String {
    "const int job=column/\{width},out=column%\{width},expert=selected_indices[job]; int local=(expert>=0&&expert<\{experts})?expert_map[expert]:-1; if(local>=\{plan.local_experts().length()}) local=-1;"
  }
  let gate = @tiled.shape_decode_body(
    inner=hidden,
    columns=selected * intermediate,
    maximum_rows=rows,
    prepare=prepare(intermediate) + addresses[0],
    left="__bfloat162float(input[inner])",
    right="(local<0?0.0f:__bfloat162float(\{symbol}_w0(w0,s0,g0,out,inner)))",
    publish="gates[column%4]=value;",
  )
  let up = @tiled.shape_decode_body(
    inner=hidden,
    columns=selected * intermediate,
    maximum_rows=rows,
    prepare=prepare(intermediate) + addresses[1],
    left="__bfloat162float(input[inner])",
    right="(local<0?0.0f:__bfloat162float(\{symbol}_w1(w1,s1,g1,out,inner)))",
    publish="__nv_bfloat16 gate=__float2bfloat16_rn(gates[column%4]),up=__float2bfloat16_rn(value); \{clamp} float g=__bfloat162float(gate),product=\{product}; \{route} activation[column]=local<0?__float2bfloat16_rn(0.0f):__float2bfloat16_rn(product);",
  )
  let publish = if plan.down_publication() == ReduceFloat32BeforeRound {
    "value"
  } else {
    @luna_cuda_precision.scalar_expression(plan.expression(DownPublish), [
      "value", "selected_weights[job]",
    ])
  }
  let down = @tiled.shape_decode_body(
    inner=intermediate,
    columns=selected * hidden,
    maximum_rows=rows,
    prepare=prepare(hidden) + addresses[2],
    left="__bfloat162float(activation[(unsigned long long)job*\{intermediate}+inner])",
    right="(local<0?0.0f:__bfloat162float(\{symbol}_w2(w2,s2,g2,out,inner)))",
    publish="weighted[column]=local<0?\{zero}:\{publish};",
  )
  (
    "if(counts[3]==1) { __shared__ float gates[4]; { \{gate} } __syncthreads(); { \{up} } return; }",
    "if(counts[3]==1) { \{down} return; }",
  )
}
