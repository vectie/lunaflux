///|
/// Preserve table-slot order and gather unbiased score weights. The startup
/// loader owns full-table validation/narrowing; no table scan enters this step.
fn hash_weight_source(
  plan : @precision.MoeRoutingPrecision,
  symbol : String,
) -> String raise {
  let (rows, _, experts, selected) = plan.geometry()
  let normalize = if plan.normalize() { "true" } else { "false" }
  let scaling = @literal.canonical_cuda_float_literal(plan.scaling())
  let sum = @arithmetic.scalar_expression(
    @precision.RoutingWeightExpression::SumUpdate.program(),
    ["sum", "value"],
  )
  let denominator = @arithmetic.scalar_expression(
    @precision.RoutingWeightExpression::Denominator.program(),
    ["sum", "1.0e-20f"],
  )
  let publish = @arithmetic.scalar_expression(
    @precision.RoutingWeightExpression::Publish.program(),
    ["scores[score_base+indices[output+slot]]", "denominator", scaling],
  )
  let reduction = if plan.selected_sum_law() is OrderedFloat32 {
    ""
  } else {
    "  if(valid) " +
    @grouped.selected_weight_butterfly_source(
      width=selected,
      value_expression="scores[score_base+indices[output+slot]]",
      output_symbol="sum",
    )
  }
  let source =
    $|extern "C" __global__ void \{symbol}(const int *counts, const float *scores, const int *indices, float *weights) {
    $|  const int live=counts[3], row=blockIdx.x;
    $|  if(live<=0||live>\{rows}||row>=live||threadIdx.x!=0) return;
    $|  const unsigned long long output=(unsigned long long)row*\{selected}ULL, score_base=(unsigned long long)row*\{experts}ULL;
    $|  float sum=0.0f; bool valid=true;
    $|  for(int slot=0;slot<\{selected};++slot) {
    $|    const int expert=indices[output+slot];
    $|    if(expert<0||expert>=\{experts}) { valid=false; continue; }
    $|    const float value=scores[score_base+expert];
    $|    if(!isfinite(value)) valid=false;
    $|    sum=\{sum};
    $|  }
    $|\{reduction}  const float denominator=\{normalize} ? \{denominator} : 1.0f;
    $|  for(int slot=0;slot<\{selected};++slot) {
    $|    if(!valid) { weights[output+slot]=0.0f; continue; }
    $|    weights[output+slot]=\{publish};
    $|  }
    $|}
  source
}
