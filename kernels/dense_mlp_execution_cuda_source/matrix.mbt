///|
pub fn DenseMlpCudaSource::launch(self : Self, stage : Int) -> (Int, Int) raise {
  guard stage is (0 | 1) ||
    (stage == 2 && self.plan.publication() == ReduceFloat32BeforeRound) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (hidden, intermediate, rows) = self.plan.geometry()
  let (inner, columns) = if stage == 0 {
    (hidden, intermediate)
  } else {
    (intermediate, hidden)
  }
  if stage < 2 && self.plan.dot_law() != ScalarFloat32 {
    let tile = @matrix.MatrixTilePlan::new(
      rows~,
      columns~,
      inner~,
      law=self.plan.dot_law(),
    )
    (tile.tile_count(), 128)
  } else {
    (((rows.to_int64() * columns.to_int64() + 255L) / 256L).to_int(), 256)
  }
}

///|
fn matrix_mlp_source(
  plan : @precision.DenseSwiGluPrecision,
  symbol : String,
) -> Bytes raise {
  let (hidden, intermediate, rows) = plan.geometry()
  let tile = @matrix.MatrixTilePlan::new(
    rows~,
    columns=intermediate,
    inner=hidden,
    law=plan.dot_law(),
  )
  let body = @tiled.paired_matrix_tile_body(
    tile,
    input="(row_base+matrix_row<live?input[(unsigned long long)(row_base+matrix_row)*\{hidden}+matrix_inner]:__float2bfloat16_rn(0.0f))",
    weights=(
      "(column_base+matrix_column<\{intermediate}?gate_weight[(unsigned long long)(column_base+matrix_column)*\{hidden}+matrix_inner]:__float2bfloat16_rn(0.0f))",
      "(column_base+matrix_column<\{intermediate}?up_weight[(unsigned long long)(column_base+matrix_column)*\{hidden}+matrix_inner]:__float2bfloat16_rn(0.0f))",
    ),
    results=("gate_result", "up_result"),
  )
  let activation = @arithmetic.scalar_expression(plan.expression(Activation), [
    "gate",
  ])
  let product = @arithmetic.scalar_expression(plan.expression(Product), [
    "activated", "up",
  ])
  let decode = if plan.dot_law() == ShapeAwareFloat32 {
    // Both sibling products consume the same shape/fold strategy. A single
    // workgroup owns each pair of output planes; no intermediate global rows.
    let gate = @tiled.shape_decode_body(
      inner=hidden,
      columns=intermediate,
      maximum_rows=rows,
      left="__bfloat162float(input[inner])",
      right="__bfloat162float(gate_weight[(unsigned long long)column*\{hidden}+inner])",
      publish="gate_result[column%4]=value;",
    )
    // The per-CTA plane is four columns because decode launch covers all N/4.
    let up = @tiled.shape_decode_body(
      inner=hidden,
      columns=intermediate,
      maximum_rows=rows,
      left="__bfloat162float(input[inner])",
      right="__bfloat162float(up_weight[(unsigned long long)column*\{hidden}+inner])",
      publish="const __nv_bfloat16 gate=__float2bfloat16_rn(gate_result[column%4]),up=__float2bfloat16_rn(value); const __nv_bfloat16 activated=\{activation}; product[column]=\{product};",
    )
    "if(live==1) { { \{gate} } __syncthreads(); { \{up} } return; }"
  } else {
    ""
  }
  let gate_source =
    $|#include <cuda_bf16.h>
    $|#include <mma.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}_gate_up(const int *counts,const __nv_bfloat16 *input,const __nv_bfloat16 *gate_weight,const __nv_bfloat16 *up_weight,__nv_bfloat16 *product) {
    $| const int live=counts[3]; if(live<=0||live>\{rows}) return;
    $| __shared__ __align__(32) float gate_result[1024],up_result[1024];
    $| \{decode}
    $| const int row_base=(blockIdx.x/\{(intermediate+63)/64})*16,column_base=(blockIdx.x%\{(intermediate+63)/64})*64;
    $| if(row_base>=live) return;
    $| \{body}
    $| for(int i=threadIdx.x;i<1024;i+=128) { int row=row_base+i/64,column=column_base+i%64;
    $|   if(row<live&&column<\{intermediate}) { const __nv_bfloat16 gate=__float2bfloat16_rn(gate_result[i]),up=__float2bfloat16_rn(up_result[i]); const __nv_bfloat16 activated=\{activation}; product[(unsigned long long)row*\{intermediate}+column]=\{product}; }
    $| }
    $|}
  @utf8.encode(gate_source) +
  b"\n" +
  @tiled.dense_source(
    @matrix.MatrixTilePlan::new(
      rows~,
      columns=hidden,
      inner=intermediate,
      law=plan.dot_law(),
    ),
    symbol=symbol + "_down",
    partial_float32=plan.publication() == ReduceFloat32BeforeRound,
  ) +
  (if plan.publication() == ReduceFloat32BeforeRound {
    b"\n" +
    @tiled.publication_source(rows~, width=hidden, symbol=symbol + "_publish")
  } else {
    b""
  })
}
