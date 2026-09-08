///|
/// Stage immutable matrix blocks while consuming the preceding block. CUDA
/// owns copy width, workgroup cooperation, and shared-address conversion.
fn projection_fold_pipeline_source(
  pipeline : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  input_base : String,
  weight_base : String,
  stride : Int,
  groups : Int,
  output_width : Int,
) -> String {
  let extent = pipeline.tile_count() * pipeline.tile_width()
  let tile = pipeline.tile_width()
  let width = pipeline.transfer_width()
  let rows = pipeline.row_width()
  let slot = (rows + groups * 16) * width
  let source =
    $|  {
    $|    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a;
    $|    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b;
    $|    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c[\{rows / 16}];
    $|    #pragma unroll
    $|    for (int row = 0; row < \{rows / 16}; ++row) nvcuda::wmma::fill_fragment(c[row], 0.0f);
    $|    __shared__ __align__(32) __nv_bfloat16 pipeline_storage[\{slot * 2}];
    $|    const __nv_bfloat16 *pipeline_a = \{input_base};
    $|    const __nv_bfloat16 *pipeline_b = \{weight_base};
    $|    auto stage_operands = [&](int offset, int stage) {
    $|      __nv_bfloat16 *destination = pipeline_storage + stage * \{slot};
    $|      for (int vector = threadIdx.x; vector < \{slot / 8}; vector += blockDim.x) {
    $|        int element = vector * 8;
    $|        bool is_a = element < \{rows * width};
    $|        int local = is_a ? element : element - \{rows * width};
    $|        int row = local / \{width}, k = local % \{width};
    $|        bool valid = !is_a || row_start + row < token_count;
    $|        const __nv_bfloat16 *source = valid ? (is_a ? pipeline_a : pipeline_b) + (unsigned long long)row * \{stride}ULL + offset + k : pipeline_a;
    $|        unsigned address = (unsigned)__cvta_generic_to_shared(destination + element);
    $|        int bytes = valid ? 16 : 0;
    $|        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(bytes) : "memory");
    $|      }
    $|      asm volatile("cp.async.commit_group;" ::: "memory");
    $|    };
    $|    stage_operands(0, 0);
    $|    asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|    __syncthreads();
    $|    int stage = 0;
    $|    #pragma unroll 1
    $|    for (int inner = 0; inner < \{extent}; inner += \{width}) {
    $|      if (inner + \{width} < \{extent}) stage_operands(inner + \{width}, stage ^ 1);
    $|      const __nv_bfloat16 *current = pipeline_storage + stage * \{slot};
    $|      #pragma unroll
    $|      for (int k = 0; k < \{width}; k += \{tile}) {
    $|        nvcuda::wmma::load_matrix_sync(b, current + \{rows * width} + warp * \{16 * width} + k, \{width});
    $|        #pragma unroll
    $|        for (int row = 0; row < \{rows / 16}; ++row) {
    $|          nvcuda::wmma::load_matrix_sync(a, current + row * \{16 * width} + k, \{width});
    $|          nvcuda::wmma::mma_sync(c[row], a, b, c[row]);
    $|        }
    $|      }
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      stage ^= 1;
    $|    }
    $|    #pragma unroll
    $|    for (int row = 0; row < \{rows / 16}; ++row) {
    $|      nvcuda::wmma::store_matrix_sync(warp_output, c[row], 16, nvcuda::wmma::mem_row_major);
    $|      __syncwarp();
    $|      #pragma unroll
    $|      for (int item = 0; item < 8; ++item) {
    $|        int element = lane + item * 32;
    $|        int output_row = row_start + row * 16 + element / 16;
    $|        if (output_row < token_count) output[(unsigned long long)output_row * \{output_width}ULL + column + element % 16] = __float2bfloat16_rn(warp_output[element]);
    $|      }
    $|      __syncwarp();
    $|    }
    $|  }
    $|
  source
}
