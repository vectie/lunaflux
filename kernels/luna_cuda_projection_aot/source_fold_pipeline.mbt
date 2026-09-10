///|
/// Stage immutable matrix blocks while consuming the preceding block. CUDA
/// owns copy width, workgroup cooperation, and shared-address conversion.
/// Empty row microtiles are unobserved members of the parallel result map,
/// not reduction terms. Their workgroup-uniform guards leave every live dot's
/// ordered updates and all inter-stage barriers unchanged.
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
  let layout : CudaProjectionTransportLayout = { columns: width }
  let input_elements = rows * width
  let slot = (rows + groups * 16) * width
  // Distribute the immutable operand product into two affine producer maps.
  // This removes the per-vector operand tag and pointer selection without
  // changing the ordered reduction or moving reads across materialization.
  // Keep the next slot's global reads independent of the current matrix fold.
  // A synchronous load/store pair blocks this producer at STS on the load's
  // scoreboard even though its destination is the inactive shared slot.
  let input_copy = projection_fold_operand_copy_source(
    "pipeline_a", "destination", rows, width, stride, true, true,
  )
  let weight_copy = projection_fold_operand_copy_source(
    "pipeline_b",
    "destination + \{input_elements}",
    groups * 16,
    width,
    stride,
    false,
    true,
  )
  let source =
    $|  {
    $|    \{layout.address_function()}
    $|    \{projection_result_address_source()}
    $|    float c[\{rows / 16}][8] = {};
    $|    __shared__ __align__(32) __nv_bfloat16 pipeline_storage[\{slot * 2}];
    $|    const __nv_bfloat16 *pipeline_a = \{input_base};
    $|    const __nv_bfloat16 *pipeline_b = \{weight_base};
    $|    auto stage_operands = [&](int offset, int stage) {
    $|      __nv_bfloat16 *destination = pipeline_storage + stage * \{slot};
    $|\{input_copy}
    $|\{weight_copy}
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
    $|        unsigned b[4];
    $|        const unsigned bp = (unsigned)__cvta_generic_to_shared(current + \{input_elements} + operand_index(warp * 16 + lane % 16, k + lane / 16 * 8));
    $|        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(bp));
    $|        #pragma unroll
    $|        for (int row = 0; row < \{rows / 16}; ++row) {
    $|          if (row_start + row * 16 >= token_count) break;
    $|          unsigned a[4];
    $|          const unsigned ap = (unsigned)__cvta_generic_to_shared(current + operand_index(row * 16 + lane % 16, k + lane / 16 * 8));
    $|          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(ap));
    $|          #pragma unroll
    $|          for (int half = 0; half < 2; ++half) {
    $|            float *d = c[row] + half * 4;
    $|            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half]), "r"(b[half+2]));
    $|          }
    $|        }
    $|      }
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      stage ^= 1;
    $|    }
    $|    #pragma unroll
    $|    for (int row = 0; row < \{rows / 16}; ++row) {
    $|      if (row_start + row * 16 >= token_count) break;
    $|      #pragma unroll
    $|      for (int item = 0; item < 8; ++item)
    $|        warp_output[result_index(lane / 4 + (item % 4 / 2) * 8, lane % 4 * 2 + item % 2 + item / 4 * 8)] = c[row][item];
    $|      __syncwarp();
    $|      #pragma unroll
    $|      for (int item = 0; item < 8; ++item) {
    $|        int element = lane + item * 32;
    $|        int output_row = row_start + row * 16 + element / 16;
    $|        if (output_row < token_count) output[(unsigned long long)output_row * \{output_width}ULL + column + element % 16] = __float2bfloat16_rn(warp_output[result_index(element / 16, element % 16)]);
    $|      }
    $|      __syncwarp();
    $|    }
    $|  }
    $|
  source
}

///|
/// Specialize one copy domain before CUDA rendering. Invalid rows are
/// zero-filled; producer and matrix consumer use one shared permutation.
fn projection_fold_operand_copy_source(
  base : String,
  destination : String,
  rows : Int,
  width : Int,
  stride : Int,
  guard_rows : Bool,
  asynchronous : Bool,
) -> String {
  let validity = if guard_rows {
    "const bool valid = row_start + row < token_count;"
  } else {
    "const bool valid = true;"
  }
  let source =
    $|      for (int vector = threadIdx.x; vector < \{rows * width / 8}; vector += blockDim.x) {
    $|        const int row = vector / \{width / 8}, k = (vector % \{width / 8}) * 8;
    $|        \{validity}
    $|        const __nv_bfloat16 *source = valid ? \{base} + (unsigned long long)row * \{stride}ULL + offset + k : \{base};
    $|        const unsigned address = (unsigned)__cvta_generic_to_shared(\{destination} + operand_index(row, k));
    $|        \{projection_operand_transfer_source(width, asynchronous~)}
    $|      }
  source
}
