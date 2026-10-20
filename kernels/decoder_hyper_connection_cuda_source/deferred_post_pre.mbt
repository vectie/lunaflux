///|
/// Fuse only a legal post -> square -> control-projection chain. The local
/// values are the published BF16 bits, not the unrounded post accumulator.
/// Normalization stays before the dot product: no algebraic reassociation.
pub fn deferred_post_pre_source(
  producer : @precision.HyperConnectionPrecision,
  consumer : @precision.HyperConnectionPrecision,
  controls_per_group~ : Int,
  symbol~ : String,
  subgroup_width? : Int = 32,
) -> Bytes raise {
  let plan = @physical.DeferredPostPrePlan::new(
    producer,
    consumer,
    controls_per_group~,
  )
  guard valid_hyper_connection_symbol(symbol) else {
    raise InvalidFunctionSymbol
  }
  let (streams, hidden, rows, workers, group) = plan.geometry()
  let flat = streams * hidden
  let controls = streams * (streams + 2)
  let groups = (controls + group - 1) / group
  let retained = plan.retained_values_per_worker()
  guard subgroup_width is (1 | 32) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let subgroup_width = if workers >= subgroup_width {
    subgroup_width
  } else {
    1
  }
  let square_tree = deferred_tree_source(workers, 1, subgroup_width)
  let dot_tree = deferred_tree_source(workers, group, subgroup_width)
  let law = producer.mix_law()
  let branch = @arithmetic.scalar_expression(law.expression(BranchProduct), [
    "post[row * \{streams} + destination]",
    "__bfloat162float(branch[row * \{hidden} + component])",
  ])
  let update = @arithmetic.scalar_expression(law.expression(ResidualUpdate), [
    "residual_sum",
    "combine[(row * \{streams} + source) * \{streams} + destination]",
    "__bfloat162float(residual[(row * \{streams} + source) * \{hidden} + component])",
  ])
  let publish = @arithmetic.scalar_expression(law.expression(Publish), [
    "branch_product", "residual_sum",
  ])
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    consumer.normalization_epsilon(),
  )
  let root = control_root_source(
    "__fadd_rn(__fdiv_rn(reduction[0][0], \{flat}.0f), \{eps})",
    consumer.control_root_law(),
  )
  let producer_body = if plan.reuses_component_operands() {
    deferred_component_source(producer, hidden, streams, workers)
  } else {
    (
      $| #pragma unroll
      $| for(int i=0;i<\{retained};++i) {
      $|   const int column=worker+i*\{workers}; float x=0.0f;
      $|   if(column<\{flat}) {
      $|     const int component=column%\{hidden}, destination=column/\{hidden};
      $|     const float branch_product=\{branch}; float residual_sum=0.0f;
      $|     #pragma unroll
      $|     for(int source=0;source<\{streams};++source) residual_sum=\{update};
      $|     const __nv_bfloat16 rounded=\{publish}; if(tile==0) output[row*\{flat}+column]=rounded;
      $|     x=__bfloat162float(rounded);
      $|   }
      $|   values[i]=x;
      $| }
    )
  }
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const int32_t *counts, const float *post, const float *combine, const __nv_bfloat16 *residual, const __nv_bfloat16 *branch, __nv_bfloat16 *output, const __nv_bfloat16 *weights, float *scratch) {
      $| const int worker=threadIdx.x, live=counts[3];
      $| if(live<=0 || live>\{rows} || blockDim.x!=\{workers} || output==residual || output==branch) return;
      $| __shared__ float reduction[\{group}][\{workers}];
      $| for(int work=blockIdx.x;work<live*\{groups};work+=gridDim.x) {
      $| const unsigned long long row=work/\{groups}; const int tile=work%\{groups};
      $| float values[\{retained}], sum=0.0f;
      $|\{producer_body}
      $| #pragma unroll
      $| for(int i=0;i<\{retained};++i) {
      $|   const float x=values[i]; if(worker+i*\{workers}<\{flat}) sum=__fadd_rn(sum,isfinite(x)?__fmul_rn(x,x):NAN);
      $| }
      $| reduction[0][worker]=sum; __syncthreads();
      $| \{square_tree}
      $| const float inverse=\{root}; if(tile==0 && worker==0) scratch[row]=inverse;
      $| if(!isfinite(inverse)) { __syncthreads(); continue; }
      $| __syncthreads(); // every inverse reader retires before dot scratch reuse
      $| float dots[\{group}]={};
      $| #pragma unroll
      $| for(int i=0;i<\{retained};++i) {
      $|   const int column=worker+i*\{workers};
      $|   if(column<\{flat}) {
      $|     const float operand=__fmul_rn(values[i],inverse);
      $|     #pragma unroll
      $|     for(int c=0;c<\{group};++c) {
      $|       const int control=tile*\{group}+c;
      $|       if(control<\{controls}) dots[c]=__fadd_rn(dots[c],__fmul_rn(operand,__bfloat162float(weights[(unsigned long long)control*\{flat}+column])));
      $|     }
      $|   }
      $| }
      $| #pragma unroll
      $| for(int c=0;c<\{group};++c) reduction[c][worker]=dots[c];
      $| __syncthreads();
      $| \{dot_tree}
      $| if(worker==0) {
      $|   #pragma unroll
      $|   for(int c=0;c<\{group};++c) if(tile*\{group}+c<\{controls}) scratch[\{rows}+row*\{controls}+tile*\{group}+c]=reduction[c][0];
      $| }
      $| __syncthreads(); // retire output readers before reusing local storage
      $| }
      $|}
    ),
  )
}
