///|
/// Lower an ordered fold over immutable operands. Gather/scatter implements
/// sparse result demand; segmented weight lookup implements Q/K/V concatenation.
fn projection_matrix_pipeline_source(
  parameters : ProjectionParameters,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  maximum_rows : Int,
) -> String {
  guard compilation.schedule().matrix_pipeline() is Some(plan) else {
    return ""
  }
  let selected = compilation.schedule().demand() == QueryRowEnds
  let rows = plan.row_width()
  let columns = plan.output_tiles() * 16
  let width = plan.transfer_width()
  let input_width = parameters.input_width
  let output_width = parameters.output_width
  let consumer_groups = compilation.schedule().output_tiles_per_workgroup()
  let transport : CudaProjectionTransportLayout = { columns: width }
  // The row-preserving permutation is compact. Both producer and consumer
  // use that map; an unrelated padded-row allocation only wastes storage.
  let slot = (rows + columns) * width
  let input_elements = rows * width
  let distribution = @luna_projection_tile_compiler.partition_projection_product_fold(
    row_tiles=rows / 16,
    output_tiles=plan.output_tiles(),
    consumer_groups~,
  ).unwrap()
  let fold_rows = distribution.rows_per_group()
  let consumers = distribution.consumer_groups()
  let row_count = if selected { "counts[2]" } else { "token_count" }
  let input_row = if selected {
    "(logical_row < pipeline_rows ? row_offsets[logical_row + 1] - 1 : -1)"
  } else {
    "logical_row"
  }
  let weight_selection = if parameters.family == QkvProjection {
    "const __nv_bfloat16 *base = query_weight; int weight_row = logical_column;\n" +
    "          if (weight_row >= \{parameters.query_width}) { weight_row -= \{parameters.query_width}; if (weight_row >= \{parameters.intermediate_width}) { weight_row -= \{parameters.intermediate_width}; base = value_weight; } else base = key_weight; }\n"
  } else {
    "const __nv_bfloat16 *base = weight; const int weight_row = logical_column;\n"
  }
  let input_copy =
    $|        \{projection_transfer_map_loop(rows * width / 8, consumer_groups * 32)} {
    $|          const int local_row = vector / \{width / 8}, k = (vector % \{width / 8}) * 8;
    $|          const int logical_row = row_begin + local_row;
    $|          const int token_row = \{input_row};
    $|          const bool valid = logical_row < pipeline_rows && token_row >= 0 && token_row < token_count;
    $|          const __nv_bfloat16 *source = valid ? input + (unsigned long long)token_row * \{input_width}ULL + k_base + k : input;
    $|          const unsigned address = (unsigned)__cvta_generic_to_shared(destination + operand_index(local_row, k));
    $|          if (overlap) {
    $|            \{projection_operand_transfer_source(width)}
    $|          } else {
    $|            \{projection_operand_transfer_source(width, asynchronous=false)}
    $|          }
    $|        }
  let weight_copy =
    $|        \{projection_transfer_map_loop(columns * width / 8, consumer_groups * 32)} {
    $|          const int local_row = vector / \{width / 8}, k = (vector % \{width / 8}) * 8;
    $|          const int logical_column = column_begin + local_row;
    $|          \{weight_selection}
    $|          const bool valid = \{if output_width % columns == 0 { "true" } else { "logical_column < " + output_width.to_string() }};
    $|          const __nv_bfloat16 *source = valid ? base + (unsigned long long)weight_row * \{input_width}ULL + k_base + k : base;
    $|          const unsigned address = (unsigned)__cvta_generic_to_shared(destination + \{input_elements} + operand_index(local_row, k));
    $|          \{projection_operand_transfer_source(width)}
    $|        }
  let fragment_declarations =
    $|      float c[\{fold_rows}][8] = {};
  // Selection changes logical row addresses, not matrix-fragment ownership.
  // Both row maps lower through the same packed transport permutation.
  let fragment_fold =
    $|            unsigned b[4];
    $|            const unsigned b_address = (unsigned)__cvta_generic_to_shared(current + \{input_elements} + operand_index(consumer_column * 16 + lane % 16, k + lane / 16 * 8));
    $|            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b[0]),"=r"(b[1]),"=r"(b[2]),"=r"(b[3]) : "r"(b_address));
    $|            #pragma unroll
    $|            for (int row = 0; row < \{fold_rows}; ++row) {
    $|              if (row_begin + (consumer_row + row) * 16 >= pipeline_rows) break;
    $|              unsigned a[4];
    $|              const unsigned a_address = (unsigned)__cvta_generic_to_shared(current + operand_index((consumer_row + row) * 16 + lane % 16, k + lane / 16 * 8));
    $|              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(a_address));
    $|              #pragma unroll
    $|              for (int half = 0; half < 2; ++half) {
    $|                float *d = c[row] + half * 4;
    $|                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half]), "r"(b[half+2]));
    $|              }
    $|            }
  let source =
    $|  const int pipeline_rows = \{row_count};
    $|  if (pipeline_rows >= \{plan.minimum_rows()}\{if selected { " && pipeline_rows <= " + maximum_rows.to_string() } else { "" }}) {
    $|    __shared__ __align__(32) __nv_bfloat16 matrix_storage[\{slot * 2}];
    $|    \{transport.address_function()}
    $|    const int column_groups = \{(output_width + columns - 1) / columns};
    $|    const int work_count = ((pipeline_rows + \{rows - 1}) / \{rows}) * column_groups;
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / column_groups) * \{rows};
    $|      const int column_begin = (work % column_groups) * \{columns};
    $|      const int consumer_row = (warp / \{distribution.output_groups()}) * \{fold_rows};
    $|      const int consumer_column = warp % \{distribution.output_groups()};
    $|      auto stage_operands = [&](int k_base, int stage, bool overlap) {
    $|        __nv_bfloat16 *destination = matrix_storage + stage * \{slot};
    $|\{input_copy}
    $|\{weight_copy}
    $|        asm volatile("cp.async.commit_group;" ::: "memory");
    $|      };
    $|\{fragment_declarations}
    $|      stage_operands(0, 0, false);
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{input_width}; inner += \{width}) {
    $|        if (inner + \{width} < \{input_width}) stage_operands(inner + \{width}, stage ^ 1, true);
    $|        if (warp < \{consumers}) {
    $|          const __nv_bfloat16 *current = matrix_storage + stage * \{slot};
    $|          #pragma unroll
    $|          for (int k = 0; k < \{width}; k += 16) {
    $|\{fragment_fold}
    $|          }
    $|        }
    $|        asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|        __syncthreads();
    $|        stage ^= 1;
    $|      }
    $|      if (warp < \{consumers}) {
    $|        #pragma unroll
    $|        for (int row = 0; row < \{fold_rows}; ++row) {
    $|          if (row_begin + (consumer_row + row) * 16 >= pipeline_rows) break;
    $|          // Every accumulator element has one owner. Store that value
    $|          // directly instead of materializing and rereading a result tile.
    $|          #pragma unroll
    $|          for (int item = 0; item < 8; ++item) {
    $|            const int logical_row = row_begin + (consumer_row + row) * 16 + lane / 4 + (item % 4 / 2) * 8;
    $|            const int token_row = \{input_row};
    $|            const int column = column_begin + consumer_column * 16 + lane % 4 * 2 + item % 2 + item / 4 * 8;
    $|            if (logical_row < pipeline_rows && token_row >= 0 && token_row < token_count && column < \{output_width}) output[(unsigned long long)token_row * \{output_width}ULL + column] = __float2bfloat16_rn(c[row][item]);
    $|          }
    $|        }
    $|      }
    $|    }
    $|    return;
    $|  }
    $|
  source
}
