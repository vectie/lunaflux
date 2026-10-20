///|
/// Private instruction selection. Four consumers retain an A fragment across
/// independent N8 accumulators. Alternate register slots allow the following
/// LDSM to issue before the current MMA. A shared slot retires after its last
/// LDSM, not after the remaining register-only matrix instructions.
pub fn fragment_pipeline_source(
  plan : @matrix.MatrixFragmentPipeline,
  symbol~ : String,
  partial_float32~ : Bool = false,
) -> Bytes {
  let matrix = plan.matrix()
  let (rows, columns, inner) = matrix.geometry()
  let (m, n, k) = plan.tile()
  let warp_columns = n / (4 / (m / 16))
  let fragments = warp_columns / 8
  let output_type = if partial_float32 { "float" } else { "__nv_bfloat16" }
  let value = if partial_float32 { "v" } else { "__float2bfloat16_rn(v)" }
  let decode = if matrix.law() == ShapeAwareFloat32 {
    let body = shape_decode_body(
      inner~,
      columns~,
      maximum_rows=rows,
      left="__bfloat162float(input[inner])",
      right="__bfloat162float(weight[(unsigned long long)column*\{inner}+inner])",
      publish="output[column]=\{if partial_float32 { "value" } else { "__float2bfloat16_rn(value)" }};",
    )
    "if(live_rows==1) {\{body}\nreturn;}"
  } else {
    ""
  }
  let copies = [
      (m, "tile_a", "input", "row_base", "live_rows"),
      (n, "tile_b", "weight", "column_base", columns.to_string()),
    ]
    .map(item => {
      let (extent, tile, pointer, base, limit) = item
      (
        $|for(int i=threadIdx.x;i<\{extent*k/8};i+=128) {
        $|  const int r=\{base}+i/8,c=copy_k+(i%8)*8;
        $|  const int valid=r<\{limit}&&c<\{inner}?16:0;
        $|  const unsigned logical=i*16;
        $|  const unsigned physical=logical^((logical>>3)&112u);
        $|  unsigned dst=(unsigned)__cvta_generic_to_shared((char*)(\{tile}+copy_slot*\{extent*k})+physical);
        $|  const __nv_bfloat16 *src=\{pointer}+(valid?(unsigned long long)r*\{inner}+c:0ULL);
        $|  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(valid):"memory");
        $|}
      )
    })
    .join("\n")
  fn load(slot : String, kk : String) -> String {
    (
      $|{
      $|  unsigned a_byte=((warp_row+lane%16)*64+\{kk}+(lane/16)*8)*2;
      $|  a_byte^=(a_byte>>3)&112u;
      $|  unsigned a_addr=(unsigned)__cvta_generic_to_shared((char*)(tile_a+slot*\{m*k})+a_byte);
      $|  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      $|    :"=r"(a[\{slot}][0]),"=r"(a[\{slot}][1]),"=r"(a[\{slot}][2]),"=r"(a[\{slot}][3]):"r"(a_addr));
      $|  #pragma unroll
      $|  for(int j=0;j<\{fragments};++j) {
      $|    unsigned b_byte=((warp_column+j*8+lane%8)*64+\{kk}+((lane/8)%2)*8)*2;
      $|    b_byte^=(b_byte>>3)&112u;
      $|    unsigned b_addr=(unsigned)__cvta_generic_to_shared((char*)(tile_b+slot*\{n*k})+b_byte);
      $|    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      $|      :"=r"(b[\{slot}][j][0]),"=r"(b[\{slot}][j][1]):"r"(b_addr));
      $|  }
      $|}
    )
  }
  fn mma(slot : String) -> String {
    (
      $|#pragma unroll
      $|for(int j=0;j<\{fragments};++j) {
      $|  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      $|    :"+f"(acc[j][0]),"+f"(acc[j][1]),"+f"(acc[j][2]),"+f"(acc[j][3])
      $|    :"r"(a[\{slot}][0]),"r"(a[\{slot}][1]),"r"(a[\{slot}][2]),"r"(a[\{slot}][3]),"r"(b[\{slot}][j][0]),"r"(b[\{slot}][j][1]));
      $|}
    )
  }
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|extern "C" __global__ void \{symbol}(const int *counts,const __nv_bfloat16 *input,const __nv_bfloat16 *weight,\{output_type} *output) {
      $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{rows})return;
      $|  \{decode}
      $|  const int row_base=(blockIdx.x/\{(columns+63)/64})*\{m},column_base=(blockIdx.x%\{(columns+63)/64})*64;
      $|  if(row_base>=live_rows)return;
      $|  __shared__ __align__(1024) __nv_bfloat16 tile_a[\{2*m*k}],tile_b[\{2*n*k}];
      $|  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
      $|  const int warp_row=(warp/\{4/(m/16)})*16,warp_column=(warp%\{4/(m/16)})*\{warp_columns};
      $|  float acc[\{fragments}][4]={};
      $|  unsigned a[2][4],b[2][\{fragments}][2];
      $|  // Prologue: two independently committed operand epochs.
      $|  for(int copy_slot=0;copy_slot<2;++copy_slot) {int copy_k=copy_slot*64;\{copies}
      $|    asm volatile("cp.async.commit_group;":::"memory");
      $|  }
      $|  for(int epoch=0;epoch<\{inner/64};++epoch) {
      $|    const int slot=epoch&1;
      $|    asm volatile("cp.async.wait_group 1;":::"memory");
      $|    __syncthreads(); // publish this epoch; the later copy stays outstanding
      $|    \{load("0", "0")}
      $|    #pragma unroll
      $|    for(int q=0;q<3;++q) {
      $|      const int next=(q+1)&1,current=q&1;
      $|      \{load("next", "(q+1)*16")}
      $|      \{mma("current")}
      $|    }
      $|    __syncthreads(); // all final fragments are in registers; retire readers
      $|    {int copy_slot=slot,copy_k=(epoch+2)*64;\{copies}}
      $|    asm volatile("cp.async.commit_group;":::"memory");
      $|    \{mma("1")}
      $|  }
      $|  asm volatile("cp.async.wait_group 0;":::"memory");
      $|  #pragma unroll
      $|  for(int j=0;j<\{fragments};++j) {
      $|    #pragma unroll
      $|    for(int t=0;t<4;++t) {
      $|      const int r=row_base+warp_row+lane/4+(t/2)*8,c=column_base+warp_column+j*8+(lane%4)*2+t%2;
      $|      float v=acc[j][t];
      $|      if(r<live_rows&&c<\{columns})output[(unsigned long long)r*\{columns}+c]=\{value};
      $|    }
      $|  }
      $|}
    ),
  )
}
