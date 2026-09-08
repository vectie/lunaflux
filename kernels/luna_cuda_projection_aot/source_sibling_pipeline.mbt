///|
/// A product of ordered folds shares both the staged input and each pair of
/// weight tiles. Copy scheduling and CUDA lane ownership are lowering details.
fn projection_sibling_fold_pipeline_source(
  plan : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  parameters : ProjectionParameters,
) -> String {
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let width = plan.transfer_width()
  let input_width = parameters.input_width
  let intermediate = parameters.intermediate_width
  let slot = (rows + 2 * columns) * width
  let source =
    $|  if (token_count >= \{plan.minimum_rows()}) {
    $|    __shared__ __align__(32) __nv_bfloat16 sibling_storage[\{slot * 2}];
    $|    extern __shared__ unsigned char sibling_output_bytes[];
    $|    float *temporary = reinterpret_cast<float *>(sibling_output_bytes) + warp * 256;
    $|    const int groups = \{intermediate / columns};
    $|    const int work_count = ((token_count + \{rows - 1}) / \{rows}) * groups;
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / groups) * \{rows};
    $|      const int column_begin = (work % groups) * \{columns};
    $|      auto stage_operands = [&](int k_base, int stage) {
    $|        __nv_bfloat16 *destination = sibling_storage + stage * \{slot};
    $|        for (int vector = threadIdx.x; vector < \{slot / 8}; vector += blockDim.x) {
    $|          const int element = vector * 8;
    $|          const bool is_input = element < \{rows * width};
    $|          const bool is_gate = element < \{(rows + columns) * width};
    $|          const int offset = is_input ? element : (is_gate ? element - \{rows * width} : element - \{(rows + columns) * width});
    $|          const int row = offset / \{width}, k = offset % \{width};
    $|          const bool valid = !is_input || row_begin + row < token_count;
    $|          const __nv_bfloat16 *base = is_input ? input : (is_gate ? gate_weight : up_weight);
    $|          const int source_row = is_input ? row_begin + row : column_begin + row;
    $|          const __nv_bfloat16 *source = valid ? base + (unsigned long long)source_row * \{input_width}ULL + k_base + k : input;
    $|          const unsigned address = (unsigned)__cvta_generic_to_shared(destination + element);
    $|          const int bytes = valid ? 16 : 0;
    $|          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(bytes) : "memory");
    $|        }
    $|        asm volatile("cp.async.commit_group;" ::: "memory");
    $|      };
    $|      using namespace nvcuda;
    $|      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    $|      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bg, bu;
    $|      wmma::fragment<wmma::accumulator, 16, 16, 16, float> cg[\{rows / 16}], cu[\{rows / 16}];
    $|      #pragma unroll
    $|      for (int row = 0; row < \{rows / 16}; ++row) { wmma::fill_fragment(cg[row], 0.0f); wmma::fill_fragment(cu[row], 0.0f); }
    $|      stage_operands(0, 0);
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{input_width}; inner += \{width}) {
    $|        if (inner + \{width} < \{input_width}) stage_operands(inner + \{width}, stage ^ 1);
    $|        if (warp < \{plan.output_tiles()}) {
    $|          const __nv_bfloat16 *current = sibling_storage + stage * \{slot};
    $|          #pragma unroll
    $|          for (int k = 0; k < \{width}; k += 16) {
    $|            wmma::load_matrix_sync(bg, current + \{rows * width} + warp * \{16 * width} + k, \{width});
    $|            wmma::load_matrix_sync(bu, current + \{(rows + columns) * width} + warp * \{16 * width} + k, \{width});
    $|            #pragma unroll
    $|            for (int row = 0; row < \{rows / 16}; ++row) {
    $|              if (row_begin + row * 16 >= token_count) break;
    $|              wmma::load_matrix_sync(a, current + row * \{16 * width} + k, \{width});
    $|              wmma::mma_sync(cg[row], a, bg, cg[row]);
    $|              wmma::mma_sync(cu[row], a, bu, cu[row]);
    $|            }
    $|          }
    $|        }
    $|        asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|        __syncthreads();
    $|        stage ^= 1;
    $|      }
    $|      if (warp < \{plan.output_tiles()}) {
    $|        #pragma unroll
    $|        for (int row = 0; row < \{rows / 16}; ++row) {
    $|          if (row_begin + row * 16 >= token_count) break;
    $|          wmma::store_matrix_sync(temporary, cg[row], 16, wmma::mem_row_major);
    $|          __syncwarp();
    $|          float gates[8];
    $|          #pragma unroll
    $|          for (int item = 0; item < 8; ++item) gates[item] = temporary[lane + item * 32];
    $|          __syncwarp();
    $|          wmma::store_matrix_sync(temporary, cu[row], 16, wmma::mem_row_major);
    $|          __syncwarp();
    $|          #pragma unroll
    $|          for (int item = 0; item < 8; ++item) {
    $|            const int element = lane + item * 32;
    $|            const int output_row = row_begin + row * 16 + element / 16;
    $|            if (output_row < token_count) {
    $|              const float gate = gates[item], up = temporary[element];
    $|              workspace[(unsigned long long)output_row * \{intermediate}ULL + column_begin + warp * 16 + element % 16] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
    $|            }
    $|          }
    $|          __syncwarp();
    $|        }
    $|      }
    $|    }
    $|    return;
    $|  }
    $|
  source
}
