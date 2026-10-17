///|
/// Row-owned fusion retains the intermediate BF16 publication. No speculative
/// reassociation: the successor's striped square fold sees exactly those bits.
pub fn post_square_source(
  producer : @precision.HyperConnectionPrecision,
  consumer : @precision.HyperConnectionPrecision,
  symbol~ : String,
) -> Bytes raise {
  let plan = @physical.PostSquareFusion::new(producer, consumer)
  guard valid_hyper_connection_symbol(symbol) else {
    raise InvalidFunctionSymbol
  }
  let (streams, hidden, rows, workers) = plan.geometry()
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
    "__fadd_rn(__fdiv_rn(squares[0], \{streams * hidden}.0f), \{eps})",
    consumer.control_root_law(),
  )
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|#include <stdint.h>
      $|extern "C" __global__ void \{symbol}(const int32_t *counts, const float *post, const float *combine, const __nv_bfloat16 *residual, const __nv_bfloat16 *branch, __nv_bfloat16 *output, float *scratch) {
      $| const unsigned long long row=blockIdx.x; const int worker=threadIdx.x;
      $| if(counts[3]<=0 || counts[3]>\{rows} || row>=(unsigned)counts[3] || blockDim.x!=\{workers} || output==residual || output==branch) return;
      $| __shared__ float squares[\{workers}]; float sum=0.0f;
      $| for(int column=worker;column<\{streams * hidden};column+=\{workers}) {
      $|   const int component=column%\{hidden}, destination=column/\{hidden};
      $|   const float branch_product=\{branch}; float residual_sum=0.0f;
      $|   for(int source=0;source<\{streams};++source) residual_sum=\{update};
      $|   const __nv_bfloat16 rounded=\{publish}; output[row*\{streams * hidden}+column]=rounded;
      $|   const float x=__bfloat162float(rounded); sum=__fadd_rn(sum,isfinite(x)?__fmul_rn(x,x):NAN);
      $| }
      $| squares[worker]=sum; __syncthreads();
      $| for(int stride=\{workers / 2};stride>0;stride>>=1) { if(worker<stride) squares[worker]=__fadd_rn(squares[worker],squares[worker+stride]); __syncthreads(); }
      $| if(worker==0) scratch[row]=\{root};
      $|}
    ),
  )
}
