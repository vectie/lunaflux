///|
/// Private dense-address lowering of the physical two-slot plan. A copy group
/// is completed before publication; the workgroup retires all readers before
/// either slot is reused. Unaligned or odd-stride operands use scalar staging
/// with the same lifetime edges and arithmetic, not an unsupported vector load.
fn buffered_dense_body(
  plan : @matrix.MatrixTilePlan,
  external_scratch~ : Bool,
) -> String {
  let (_, columns, inner) = plan.geometry()
  let (m, n, k) = plan.tile()
  let scratch = if external_scratch {
    ""
  } else {
    "__shared__ __align__(32) __nv_bfloat16 tile_a[\{2*m*k}],tile_b[\{2*n*k}];"
  }
  let copies = [
      (m, "tile_a", "input", "row_base", "live_rows"),
      (n, "tile_b", "weight", "column_base", columns.to_string()),
    ]
    .map(args => {
      let (extent, tile, pointer, row_base, limit) = args
      let scalar =
        $|for(int i=threadIdx.x;i<\{extent*k};i+=128) {
        $|  int r=\{row_base}+i/\{k},c=copy_k+i%\{k};
        $|  \{tile}[copy_slot*\{extent*k}+i]=r<\{limit}&&c<\{inner}?\{pointer}[(unsigned long long)r*\{inner}+c]:__float2bfloat16_rn(0.0f);
        $|}
      if inner % 8 != 0 {
        scalar
      } else {
        (
          $|if((((unsigned long long)\{pointer})&15ULL)==0ULL) {
          $|  for(int i=threadIdx.x;i<\{extent*k/8};i+=128) {
          $|    int r=\{row_base}+i/\{k/8},c=copy_k+(i%\{k/8})*8;
          $|    int bytes=r<\{limit}&&c<\{inner}?16:0;
          $|    const __nv_bfloat16 *src=\{pointer}+(bytes?(unsigned long long)r*\{inner}+c:0ULL);
          $|    unsigned dst=(unsigned)__cvta_generic_to_shared(\{tile}+copy_slot*\{extent*k}+i*8);
          $|    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst),"l"(src),"r"(bytes):"memory");
          $|  }
          $|} else { \{scalar} }
        )
      }
    })
    .join("\n")
  (
    $|{ // physical operand ring=2, K=\{k}; ascending K16 numerical fold
    $|  \{scratch}
    $|  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> accumulator;
    $|  nvcuda::wmma::fill_fragment(accumulator,0.0f);
    $|  const int warp=threadIdx.x/32;
    $|  {int copy_k=0,copy_slot=0;\{copies}}
    $|  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    $|  __syncthreads();
    $|  for(int base_k=0,epoch=0;base_k<\{inner};base_k+=\{k},++epoch) {
    $|    int slot=epoch&1;
    $|    if(base_k+\{k}<\{inner}) {int copy_k=base_k+\{k},copy_slot=slot^1;\{copies}}
    $|    asm volatile("cp.async.commit_group;" ::: "memory");
    $|    #pragma unroll
    $|    for(int kk=0;kk<\{k};kk+=16) {
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> a;
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> b;
    $|      nvcuda::wmma::load_matrix_sync(a,tile_a+slot*\{m*k}+kk,\{k});
    $|      nvcuda::wmma::load_matrix_sync(b,tile_b+slot*\{n*k}+warp*16*\{k}+kk,\{k});
    $|      nvcuda::wmma::mma_sync(accumulator,a,b,accumulator);
    $|    }
    $|    asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|    __syncthreads(); // publish next slot and retire current readers
    $|  }
    $|  nvcuda::wmma::store_matrix_sync(result+warp*16,accumulator,\{n},nvcuda::wmma::mem_row_major);
    $|  __syncthreads();
    $|}
  )
}
