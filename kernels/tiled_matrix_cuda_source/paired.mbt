///|
/// Two projections of the same input tile. The immutable matrix plan supplies
/// arithmetic/geometry; this private device lowering shares operand staging
/// and publication barriers while retaining independent F32 accumulators.
pub fn paired_matrix_tile_body(
  plan : @matrix.MatrixTilePlan,
  input~ : String,
  weights~ : (String, String),
  results~ : (String, String),
  input_vector? : String,
  weight_vectors? : (String, String),
  vector_guard? : String = "true",
) -> String raise @precision.PrecisionPlanError {
  guard plan.staging() == Synchronous else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (_, _, inner) = plan.geometry()
  let (m, n, k) = plan.tile()
  let scalar_a =
    $|for(int i=threadIdx.x;i<\{m*k};i+=128) {
    $|  const int matrix_row=i/\{k},matrix_inner=base_k+i%\{k};
    $|  tile_a[i]=matrix_inner<\{inner}?(\{input}):__float2bfloat16_rn(0.0f);
    $|}
  let load_a = match input_vector {
    None => scalar_a
    Some(load) =>
      (
        $|if(\{vector_guard}) { for(int i=threadIdx.x;i<\{m*k/8};i+=128) {
        $|  const int matrix_row=i/\{k/8},matrix_inner=base_k+(i%\{k/8})*8;
        $|  reinterpret_cast<uint4 *>(tile_a)[i]=matrix_inner<\{inner}?(\{load}):make_uint4(0,0,0,0);
        $|} } else { \{scalar_a} }
      )
  }
  let scalar_b =
    $|for(int i=threadIdx.x;i<\{n*k};i+=128) {
    $|  const int matrix_column=i/\{k},matrix_inner=base_k+i%\{k};
    $|  tile_b0[i]=matrix_inner<\{inner}?(\{weights.0}):__float2bfloat16_rn(0.0f);
    $|  tile_b1[i]=matrix_inner<\{inner}?(\{weights.1}):__float2bfloat16_rn(0.0f);
    $|}
  let load_b = match weight_vectors {
    None => scalar_b
    Some(loads) =>
      (
        $|if(\{vector_guard}) { for(int i=threadIdx.x;i<\{n*k/8};i+=128) {
        $|  const int matrix_column=i/\{k/8},matrix_inner=base_k+(i%\{k/8})*8;
        $|  reinterpret_cast<uint4 *>(tile_b0)[i]=matrix_inner<\{inner}?(\{loads.0}):make_uint4(0,0,0,0);
        $|  reinterpret_cast<uint4 *>(tile_b1)[i]=matrix_inner<\{inner}?(\{loads.1}):make_uint4(0,0,0,0);
        $|} } else { \{scalar_b} }
      )
  }
  (
    $|{
    $|  __shared__ __align__(32) __nv_bfloat16 tile_a[\{m*k}],tile_b0[\{n*k}],tile_b1[\{n*k}];
    $|  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc0,acc1;
    $|  nvcuda::wmma::fill_fragment(acc0,0.0f); nvcuda::wmma::fill_fragment(acc1,0.0f);
    $|  const int warp=threadIdx.x/32;
    $|  for(int base_k=0;base_k<\{inner};base_k+=\{k}) {
    $|    \{load_a}
    $|    \{load_b}
    $|    __syncthreads();
    $|    #pragma unroll
    $|    for(int kk=0;kk<\{k};kk+=16) {
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> a;
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> b0,b1;
    $|      nvcuda::wmma::load_matrix_sync(a,tile_a+kk,\{k});
    $|      nvcuda::wmma::load_matrix_sync(b0,tile_b0+warp*16*\{k}+kk,\{k});
    $|      nvcuda::wmma::load_matrix_sync(b1,tile_b1+warp*16*\{k}+kk,\{k});
    $|      nvcuda::wmma::mma_sync(acc0,a,b0,acc0);
    $|      nvcuda::wmma::mma_sync(acc1,a,b1,acc1);
    $|    }
    $|    __syncthreads();
    $|  }
    $|  nvcuda::wmma::store_matrix_sync(\{results.0}+warp*16,acc0,\{n},nvcuda::wmma::mem_row_major);
    $|  nvcuda::wmma::store_matrix_sync(\{results.1}+warp*16,acc1,\{n},nvcuda::wmma::mem_row_major);
    $|  __syncthreads();
    $|}
  )
}
