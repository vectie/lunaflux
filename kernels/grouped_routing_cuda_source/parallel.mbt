///|
/// Parallel map of independent group choices followed by associative stable
/// maximum selection. Group sums and selected-weight arithmetic keep the
/// existing numerical order. This is a physical lowering, not a new score law.
pub fn GroupedRoutingCudaSource::parallel_source_bytes(self : Self) -> Bytes {
  let experts = self.routed_expert_count
  let groups = self.group_count
  let per_group = experts / groups
  let selected = self.experts_per_token
  let top = self.group_score_topk
  let selected_groups = self.selected_group_count
  let group_sum = weight_expression(SumUpdate, ["sum", "best_value"])
  let selected_sum = weight_expression(SumUpdate, [
    "denominator", "uncorrected_scores[score_base+selected_indices[output_base+slot]]",
  ])
  let denominator = weight_expression(Denominator, ["denominator", "1.0e-20f"])
  let scale = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    self.scaling,
  )
  let publish = weight_expression(Publish, ["value", "denominator", scale])
  let unnormalized = weight_expression(ScaleOnly, ["value", scale])
  let reduction = if self.numerical_contract ==
    SerialF32TopGroupChoiceButterflyGatherNormalizeScaleV1 {
    try! selected_weight_butterfly_source(
      width=selected,
      value_expression="uncorrected_scores[score_base+selected_indices[output_base+slot]]",
      output_symbol="denominator",
    )
  } else {
    ""
  }
  let source =
    $|#include <math.h>
    $|#include <stdint.h>
    $|extern "C" __global__ void \{self.function_symbol}(const int *counts,const float *uncorrected_scores,const float *choice_scores,int *selected_indices,float *selected_weights) {
    $| const int row=blockIdx.x,lane=threadIdx.x,live=counts[3];
    $| if(live<0||live>\{self.maximum_rows}||row>=live) return;
    $| const unsigned long long score_base=(unsigned long long)row*\{experts},output_base=(unsigned long long)row*\{selected};
    $| __shared__ float group_scores[\{groups}],values[128];
    $| __shared__ int chosen_groups[\{selected_groups}],ids[128],bad[128],chosen[\{selected}];
    $| int invalid=0;
    $| for(int e=lane;e<\{experts};e+=128) if(!isfinite(uncorrected_scores[score_base+e])||!isfinite(choice_scores[score_base+e])) invalid=1;
    $| bad[lane]=invalid; __syncthreads();
    $| for(int span=64;span>0;span>>=1) {if(lane<span) bad[lane]|=bad[lane+span];__syncthreads();}
    $| if(bad[0]) {for(int slot=lane;slot<\{selected};slot+=128) {selected_indices[output_base+slot]=-1;selected_weights[output_base+slot]=0.0f;}return;}
    $| if(lane<\{groups}) {int previous[\{top}];float sum=0.0f;
    $|  for(int slot=0;slot<\{top};++slot) {int best=-1;float best_value=-INFINITY;
    $|   for(int i=0;i<\{per_group};++i) {int e=lane*\{per_group}+i;bool used=false;for(int j=0;j<slot;++j) if(previous[j]==e) used=true;float value=choice_scores[score_base+e];if(!used&&(best<0||value>best_value||(value==best_value&&e<best))) {best=e;best_value=value;}}
    $|   previous[slot]=best;sum=\{group_sum};
    $|  } group_scores[lane]=sum;
    $| } __syncthreads();
    $| if(lane==0) for(int slot=0;slot<\{selected_groups};++slot) {int best=-1;float best_value=-INFINITY;
    $|  for(int g=0;g<\{groups};++g) {bool used=false;for(int j=0;j<slot;++j) if(chosen_groups[j]==g) used=true;float value=group_scores[g];if(!used&&(best<0||value>best_value||(value==best_value&&g<best))) {best=g;best_value=value;}}chosen_groups[slot]=best;
    $| } __syncthreads();
    $| for(int slot=0;slot<\{selected};++slot) {int best=-1;float best_value=-INFINITY;
    $|  for(int e=lane;e<\{experts};e+=128) {bool group_ok=false,used=false;for(int j=0;j<\{selected_groups};++j) if(chosen_groups[j]==e/\{per_group}) group_ok=true;for(int j=0;j<slot;++j) if(chosen[j]==e) used=true;float value=choice_scores[score_base+e];if(group_ok&&!used&&(best<0||value>best_value||(value==best_value&&e<best))) {best=e;best_value=value;}}
    $|  values[lane]=best_value;ids[lane]=best;__syncthreads();
    $|  for(int span=64;span>0;span>>=1) {if(lane<span) {int e=ids[lane+span];float value=values[lane+span];if(e>=0&&(ids[lane]<0||value>values[lane]||(value==values[lane]&&e<ids[lane]))) {ids[lane]=e;values[lane]=value;}}__syncthreads();}
    $|  if(lane==0) {chosen[slot]=ids[0];selected_indices[output_base+slot]=ids[0];}__syncthreads();
    $| }
    $| if(lane==0) {float denominator=0.0f;for(int slot=0;slot<\{selected};++slot) denominator=\{selected_sum};\{reduction}
    $|  if(\{self.normalize}) denominator=\{denominator};
    $|  for(int slot=0;slot<\{selected};++slot) {float value=uncorrected_scores[score_base+selected_indices[output_base+slot]];selected_weights[output_base+slot]=\{self.normalize}?\{publish}:\{unnormalized};}
    $| }
    $|}
  @utf8.encode(source)
}
