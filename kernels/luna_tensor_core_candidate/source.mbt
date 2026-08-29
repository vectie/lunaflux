///|
fn tensor_core_wmma_source(entry_point : String, tile_count : Int) -> Bytes {
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <mma.h>\n#include <stdint.h>\n" +
    "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200\n" +
    "#error \"LunaTile tensor-core candidate requires exact sm120\"\n#endif\n" +
    "extern \"C\" __global__ void " +
    entry_point +
    "(const __nv_bfloat16* __restrict__ left, " +
    "const __nv_bfloat16* __restrict__ right, " +
    "const float* __restrict__ accumulator, " +
    "float* __restrict__ output) {\n" +
    "  if (gridDim.x != " +
    tile_count.to_string() +
    " || gridDim.y != 1 || gridDim.z != 1 || " +
    "blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1) return;\n" +
    "  const unsigned long long tile = (unsigned long long)blockIdx.x;\n" +
    "  const unsigned long long element_offset = tile * 256ULL;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, " +
    "__nv_bfloat16, nvcuda::wmma::row_major> left_fragment;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, " +
    "__nv_bfloat16, nvcuda::wmma::row_major> right_fragment;\n" +
    "  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, " +
    "float> accumulator_fragment;\n" +
    "  nvcuda::wmma::load_matrix_sync(left_fragment, left + element_offset, 16);\n" +
    "  nvcuda::wmma::load_matrix_sync(right_fragment, right + element_offset, 16);\n" +
    "  nvcuda::wmma::load_matrix_sync(accumulator_fragment, accumulator, 16, " +
    "nvcuda::wmma::mem_row_major);\n" +
    "  nvcuda::wmma::mma_sync(accumulator_fragment, left_fragment, " +
    "right_fragment, accumulator_fragment);\n" +
    "  nvcuda::wmma::store_matrix_sync(output + element_offset, " +
    "accumulator_fragment, 16, nvcuda::wmma::mem_row_major);\n" +
    "}\n",
  )
}
