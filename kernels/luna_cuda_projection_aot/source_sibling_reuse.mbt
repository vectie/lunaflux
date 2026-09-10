///|
/// Lower the independent sibling axis to consumers, sharing each weight
/// fragment across its row product. Once the ordered fold finishes, its input
/// staging lifetime ends; reuse that storage for the gate epilogue. The up
/// epilogue uses the already admitted dynamic scratch. CTA fences delimit both
/// changes of ownership, including inactive tail-row consumers.
/// CUDA lowering uses the documented m16n8k16 register layout, not the opaque
/// WMMA fragment representation. Two independent column halves retain the same
/// ordered K16 fold. Row-contiguous swizzled operands feed ldmatrix; register-major
/// result planes make each warp's shared stores and subsequent loads contiguous.
fn projection_sibling_reuse_source(
  plan : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  parameters : ProjectionParameters,
  reuse : @luna_projection_tile_compiler.ProjectionProductFoldDistribution,
  groups~ : Int,
) -> String {
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let width = plan.transfer_width()
  let layout : CudaProjectionTransportLayout = { columns: width }
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
    $|    \{layout.address_function()}
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / groups) * \{rows};
    $|      const int column_begin = (work % groups) * \{columns};
    $|      \{projection_sibling_phased_transfer_source(rows, columns, width, input_width, groups * 32)}
    $|      float c[\{folds}][8] = {};
    $|      stage_operands(0, 0);
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{input_width}; inner += \{width}) {
    $|        if (warp < \{consumers * 2}) {
    $|          const __nv_bfloat16 *current = sibling_storage + stage * \{slot};
    $|          #pragma unroll
    $|          for (int k = 0; k < \{width}; k += 16) {
    $|            unsigned b[4];
    $|            const unsigned bp = (unsigned)__cvta_generic_to_shared(current + \{layout.offset(rows, 0)} + sibling * \{columns * width} + operand_index(consumer_column * 16 + lane % 16, k + (lane / 16) * 8));
    $|            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(bp));
    $|            #pragma unroll
    $|            for (int row = 0; row < \{folds}; ++row) {
    $|              if (row_begin + (consumer_row + row) * 16 >= token_count) break;
    $|              unsigned a[4];
    $|              const unsigned ap = (unsigned)__cvta_generic_to_shared(current + operand_index((consumer_row + row) * 16 + lane % 16, k + (lane / 16) * 8));
    $|              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(ap));
    $|              #pragma unroll
    $|              for (int half = 0; half < 2; ++half) {
    $|                float *d = c[row] + half * 4;
    $|                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half]), "r"(b[half+2]));
    $|              }
    $|            }
    $|          }
    $|        }
    $|        if (inner + \{width} < \{input_width}) stage_operands(inner + \{width}, stage ^ 1);
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
    $|          #pragma unroll
    $|          for (int value = 0; value < 8; ++value)
    $|            result_tiles[((consumer_row + row) * \{reuse.output_groups()} + consumer_column) * 256 + value * 32 + lane] = c[row][value];
    $|        }
    $|      }
    $|      __syncthreads();
    $|      for (int element = threadIdx.x; element < \{rows * columns}; element += blockDim.x) {
    $|        const int tile = element / 256, cell = element % 256;
    $|        const int value = cell / 32, owner_lane = cell % 32;
    $|        const int output_row = row_begin + (tile / \{reuse.output_groups()}) * 16 + owner_lane / 4 + ((value % 4) / 2) * 8;
    $|        if (output_row < token_count) {
    $|          const float gate = gate_tiles[element], up = up_tiles[element];
    $|          workspace[(unsigned long long)output_row * \{intermediate}ULL + column_begin + (tile % \{reuse.output_groups()}) * 16 + (owner_lane % 4) * 2 + value % 2 + (value / 4) * 8] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
    $|        }
    $|      }
    $|      __syncthreads();
    $|    }
    $|    return;
    $|  }
    $|
  source
}
