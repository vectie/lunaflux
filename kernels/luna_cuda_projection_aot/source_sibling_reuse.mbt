///|
/// Lower the independent sibling axis to consumers, sharing each weight
/// fragment across its row product. Once the ordered fold finishes, its input
/// staging lifetime ends; reuse that storage for the gate epilogue. The up
/// epilogue uses the already admitted dynamic scratch. CTA fences delimit both
/// changes of ownership, including inactive tail-row consumers.
fn projection_sibling_reuse_source(
  plan : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  parameters : ProjectionParameters,
  reuse : @luna_projection_tile_compiler.ProjectionProductFoldDistribution,
) -> String {
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let width = plan.transfer_width()
  let slot = (rows + 2 * columns) * width
  let folds = reuse.rows_per_group()
  let consumers = reuse.consumer_groups()
  let input_width = parameters.input_width
  let intermediate = parameters.intermediate_width
  let source =
    $|  if (token_count >= \{plan.minimum_rows()}) {
    $|    __shared__ __align__(32) __nv_bfloat16 sibling_storage[\{slot * 2}];
    $|    extern __shared__ unsigned char sibling_output_bytes[];
    $|    const int groups = \{intermediate / columns};
    $|    const int work_count = ((token_count + \{rows - 1}) / \{rows}) * groups;
    $|    const int sibling = warp / \{consumers};
    $|    const int consumer = warp % \{consumers};
    $|    const int consumer_row = (consumer / \{reuse.output_groups()}) * \{folds};
    $|    const int consumer_column = consumer % \{reuse.output_groups()};
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
    $|      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    $|      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[\{folds}];
    $|      #pragma unroll
    $|      for (int row = 0; row < \{folds}; ++row) wmma::fill_fragment(c[row], 0.0f);
    $|      stage_operands(0, 0);
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{input_width}; inner += \{width}) {
    $|        if (inner + \{width} < \{input_width}) stage_operands(inner + \{width}, stage ^ 1);
    $|        if (warp < \{consumers * 2}) {
    $|          const __nv_bfloat16 *current = sibling_storage + stage * \{slot};
    $|          #pragma unroll
    $|          for (int k = 0; k < \{width}; k += 16) {
    $|            wmma::load_matrix_sync(b, current + \{rows * width} + sibling * \{columns * width} + consumer_column * \{16 * width} + k, \{width});
    $|            #pragma unroll
    $|            for (int row = 0; row < \{folds}; ++row) {
    $|              if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|              wmma::load_matrix_sync(a, current + (consumer_row + row) * \{16 * width} + k, \{width});
    $|              wmma::mma_sync(c[row], a, b, c[row]);
    $|            }
    $|          }
    $|        }
    $|        asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|        __syncthreads();
    $|        stage ^= 1;
    $|      }
    $|      float *gate_tiles = reinterpret_cast<float *>(sibling_storage);
    $|      float *up_tiles = reinterpret_cast<float *>(sibling_output_bytes);
    $|      if (warp < \{consumers * 2}) {
    $|        float *result_tiles = sibling == 0 ? gate_tiles : up_tiles;
    $|        #pragma unroll
    $|        for (int row = 0; row < \{folds}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|          wmma::store_matrix_sync(result_tiles + ((consumer_row + row) * \{reuse.output_groups()} + consumer_column) * 256, c[row], 16, wmma::mem_row_major);
    $|        }
    $|      }
    $|      __syncthreads();
    $|      for (int element = threadIdx.x; element < \{rows * columns}; element += blockDim.x) {
    $|        const int tile = element / 256, cell = element % 256;
    $|        const int output_row = row_begin + (tile / \{reuse.output_groups()}) * 16 + cell / 16;
    $|        if (output_row < token_count) {
    $|          const float gate = gate_tiles[element], up = up_tiles[element];
    $|          workspace[(unsigned long long)output_row * \{intermediate}ULL + column_begin + (tile % \{reuse.output_groups()}) * 16 + cell % 16] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
    $|        }
    $|      }
    $|      __syncthreads();
    $|    }
    $|    return;
    $|  }
    $|
  source
}
