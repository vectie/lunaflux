///|
/// Private CUDA lowering shared by dense and indirect grouped matrices.
/// Every thread must enter this body. Load expressions bind matrix_row,
/// matrix_column and matrix_inner; callers supply zero for masked operands.
/// Result is a CTA-shared F32 [16,64] array. There is no model selection here.
pub fn matrix_tile_body(
  plan : @matrix.MatrixTilePlan,
  input~ : String,
  weight~ : String,
  result~ : String,
  input_vector? : String,
  weight_vector? : String,
  vector_guard? : String = "true",
  external_scratch? : Bool = false,
) -> String raise @precision.PrecisionPlanError {
  // Expression operands have no address/extent contract for asynchronous
  // copies. Never silently discard an explicitly requested staging policy.
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
  let scalar_b =
    $|for(int i=threadIdx.x;i<\{n*k};i+=128) {
    $|  const int matrix_column=i/\{k},matrix_inner=base_k+i%\{k};
    $|  tile_b[i]=matrix_inner<\{inner}?(\{weight}):__float2bfloat16_rn(0.0f);
    $|}
  let load_a = match input_vector {
    Some(load) =>
      (
        $|if(\{vector_guard}) { for(int i=threadIdx.x;i<\{m*k/8};i+=128) {
        $|  const int matrix_row=i/\{k/8},matrix_inner=base_k+(i%\{k/8})*8;
        $|  reinterpret_cast<uint4 *>(tile_a)[i]=matrix_inner<\{inner}?(\{load}):make_uint4(0,0,0,0);
        $|} } else { \{scalar_a} }
      )
    None => scalar_a
  }
  let load_b = match weight_vector {
    Some(load) =>
      (
        $|if(\{vector_guard}) { for(int i=threadIdx.x;i<\{n*k/8};i+=128) {
        $|  const int matrix_column=i/\{k/8},matrix_inner=base_k+(i%\{k/8})*8;
        $|  reinterpret_cast<uint4 *>(tile_b)[i]=matrix_inner<\{inner}?(\{load}):make_uint4(0,0,0,0);
        $|} } else { \{scalar_b} }
      )
    None => scalar_b
  }
  let scratch = if external_scratch {
    ""
  } else {
    "__shared__ __align__(32) __nv_bfloat16 tile_a[\{m*k}], tile_b[\{n*k}];"
  }
  (
    $|{
    $|  \{scratch}
    $|  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> accumulator;
    $|  nvcuda::wmma::fill_fragment(accumulator,0.0f);
    $|  const int warp=threadIdx.x/32;
    $|  for(int base_k=0;base_k<\{inner};base_k+=\{k}) {
    $|    \{load_a}
    $|    \{load_b}
    $|    __syncthreads();
    $|    #pragma unroll
    $|    for(int kk=0;kk<\{k};kk+=16) {
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> a;
    $|      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> b;
    $|      nvcuda::wmma::load_matrix_sync(a,tile_a+kk,\{k});
    $|      nvcuda::wmma::load_matrix_sync(b,tile_b+warp*16*\{k}+kk,\{k});
    $|      nvcuda::wmma::mma_sync(accumulator,a,b,accumulator);
    $|    }
    $|    __syncthreads();
    $|  }
    $|  nvcuda::wmma::store_matrix_sync(\{result}+warp*16,accumulator,\{n},nvcuda::wmma::mem_row_major);
    $|  __syncthreads();
    $|}
  )
}

///|
pub fn dense_source(
  plan : @matrix.MatrixTilePlan,
  symbol~ : String,
  partial_float32? : Bool = false,
) -> Bytes raise @precision.PrecisionPlanError {
  @utf8.encode(
    dense_render(plan, symbol~, partial_float32~, device_region=false),
  )
}

///|
fn dense_render(
  plan : @matrix.MatrixTilePlan,
  symbol~ : String,
  partial_float32~ : Bool,
  device_region~ : Bool,
) -> String raise @precision.PrecisionPlanError {
  let (rows, columns, inner) = plan.geometry()
  let tile_id = if device_region { "tile_id" } else { "blockIdx.x" }
  let tile_count = if device_region { "tile_count" } else { "gridDim.x" }
  let blocks_n = (columns + 63) / 64
  let body = if plan.operand_slots() == 2 {
    buffered_dense_body(plan, external_scratch=device_region)
  } else {
    matrix_tile_body(
      plan,
      external_scratch=device_region,
      input="(row_base+matrix_row<live_rows?input[(unsigned long long)(row_base+matrix_row)*\{inner}+matrix_inner]:__float2bfloat16_rn(0.0f))",
      weight="(column_base+matrix_column<\{columns}?weight[(unsigned long long)(column_base+matrix_column)*\{inner}+matrix_inner]:__float2bfloat16_rn(0.0f))",
      result="result",
      vector_guard="(((unsigned long long)input|(unsigned long long)weight)&15ULL)==0ULL",
      input_vector?=if inner % 8 == 0 {
        Some(
          "(row_base+matrix_row<live_rows?*reinterpret_cast<const uint4 *>(input+(unsigned long long)(row_base+matrix_row)*\{inner}+matrix_inner):make_uint4(0,0,0,0))",
        )
      } else {
        None
      },
      weight_vector?=if inner % 8 == 0 {
        Some(
          "(column_base+matrix_column<\{columns}?*reinterpret_cast<const uint4 *>(weight+(unsigned long long)(column_base+matrix_column)*\{inner}+matrix_inner):make_uint4(0,0,0,0))",
        )
      } else {
        None
      },
    )
  }
  let output_type = if partial_float32 { "float" } else { "__nv_bfloat16" }
  let output_value = if partial_float32 {
    "result[i]"
  } else {
    "__float2bfloat16_rn(result[i])"
  }
  let decode = if plan.law() == ShapeAwareFloat32 {
    let body = shape_decode_body(
      inner~,
      columns~,
      maximum_rows=rows,
      tile_id~,
      tile_count~,
      left="__bfloat162float(input[inner])",
      right="__bfloat162float(weight[(unsigned long long)column*\{inner}+inner])",
      publish="output[column]=\{if partial_float32 { "value" } else { "__float2bfloat16_rn(value)" }};",
    )
    "if(live_rows==1) { \{body}\n return; }"
  } else {
    ""
  }
  let qualifier = if device_region {
    "__device__ __forceinline__"
  } else {
    "extern \"C\" __global__"
  }
  let coordinates = if device_region {
    "int tile_id,int tile_count,__nv_bfloat16 *tile_a,__nv_bfloat16 *tile_b,float *result,"
  } else {
    ""
  }
  let result_scratch = if device_region {
    ""
  } else {
    "__shared__ __align__(32) float result[1024];"
  }
  let text =
    $|#include <cuda_bf16.h>
    $|#include <mma.h>
    $|\{qualifier} void \{symbol}(\{coordinates}const int *counts,const __nv_bfloat16 *input,const __nv_bfloat16 *weight,\{output_type} *output) {
    $|  const int live_rows=counts[3]; if(live_rows<=0||live_rows>\{rows}) return;
    $|  \{decode}
    $|  const int row_base=(\{tile_id}/\{blocks_n})*16,column_base=(\{tile_id}%\{blocks_n})*64;
    $|  if(row_base>=live_rows) return;
    $|  \{result_scratch}
    $|  \{body}
    $|  for(int i=threadIdx.x;i<1024;i+=128) {
    $|    int row=row_base+i/64,column=column_base+i%64;
    $|    if(row<live_rows&&column<\{columns}) output[(unsigned long long)row*\{columns}+column]=\{output_value};
    $|  }
    $|}
  text
}
