///|
/// Preserve table-slot order and gather unbiased score weights. The startup
/// loader owns full-table validation/narrowing; no table scan enters this step.
fn hash_weight_source(
  plan : @precision.MoeRoutingPrecision,
  symbol : String,
) -> String {
  let (rows, _, experts, selected) = plan.geometry()
  let normalize = if plan.normalize() { "true" } else { "false" }
  let scaling = @literal.canonical_cuda_float_literal(plan.scaling())
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
    $|    sum=__fadd_rn(sum,value);
    $|  }
    $|  const float denominator=\{normalize} ? __fadd_rn(sum,1.0e-20f) : 1.0f;
    $|  for(int slot=0;slot<\{selected};++slot) {
    $|    if(!valid) { weights[output+slot]=0.0f; continue; }
    $|    weights[output+slot]=__fmul_rn(__fdiv_rn(scores[score_base+indices[output+slot]],denominator),\{scaling});
    $|  }
    $|}
  source
}
