///|
/// Device-private ownership for a strided head matrix. Reuses the same matrix
/// primitive as dense/grouped projection; no checkpoint-specific ABI here.
pub fn head_projection_source(
  plan : @matrix.HeadProjectionPlan,
  symbol~ : String,
) -> Bytes raise @precision.PrecisionPlanError {
  let tile = plan.tile()
  let (rows, columns, inner) = tile.geometry()
  let (heads, weight_heads) = plan.heads()
  let (hs, cs, ks, offset) = plan.strides()
  let body = matrix_tile_body(
    tile,
    input="(row_base+matrix_row<live?input[((uint64_t)(row_base+matrix_row)*\{heads}+head)*\{inner}+matrix_inner]:__float2bfloat16_rn(0.0f))",
    weight="(column_base+matrix_column<\{columns}?weight[weight_base+(uint64_t)(column_base+matrix_column)*\{cs}ULL+(uint64_t)matrix_inner*\{ks}ULL]:__float2bfloat16_rn(0.0f))",
    result="result",
    weight_column_first=cs == 1L && ks > 1L,
  )
  let source =
    $|#include <cuda_bf16.h>
    $|#include <mma.h>
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbol}(const int32_t *counts,const __nv_bfloat16 *input,const __nv_bfloat16 *weight,__nv_bfloat16 *output) {
    $| const int head=blockIdx.x%\{heads},tile_id=blockIdx.x/\{heads};
    $| const int row_base=(tile_id/\{(columns+63)/64})*16,column_base=(tile_id%\{(columns+63)/64})*64;
    $| const int live=counts[3];if(live<0||live>\{rows})return;
    $| if(row_base>=live){
    $|  for(int i=threadIdx.x;i<16*64;i+=128){const int row=row_base+i/64,column=column_base+i%64;if(row<\{rows}&&column<\{columns})output[((uint64_t)row*\{heads}+head)*\{columns}+column]=__float2bfloat16_rn(0.0f);}
    $|  return;
    $| }
    $| const uint64_t weight_base=(uint64_t)(head/\{heads/weight_heads})*\{hs}ULL+\{offset}ULL;
    $| __shared__ __align__(32) float result[16*64];
    $| \{body}
    $| for(int i=threadIdx.x;i<16*64;i+=128) {
    $|  const int row=row_base+i/64,column=column_base+i%64;
    $|  if(row<\{rows}&&column<\{columns})output[((uint64_t)row*\{heads}+head)*\{columns}+column]=__float2bfloat16_rn(row<live?result[i]:0.0f);
    $| }
    $|}
  @utf8.encode(source)
}
