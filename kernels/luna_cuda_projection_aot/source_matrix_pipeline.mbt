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
  let legacy_selected = selected && width < 64
  let input_width = parameters.input_width
  let output_width = parameters.output_width
  let consumer_groups = compilation.schedule().output_tiles_per_workgroup()
  // The selected-row fallback shares this launch and reserves its input tile
  // in addition to the per-consumer accumulator arena (16x16 BF16 = 512 B).
  let selected_input_bytes = if selected {
    rows * plan.tile_width() * 2
  } else {
    0
  }
  // This kernel uses static operand storage plus the existing dynamic output
  // arena. Do not add padding across the CUDA default per-block limit; wider
  // externally tuned launches retain their original physical row layout.
  let padded_bytes = (rows + columns) * (width + 16) * 2 * 2 +
    consumer_groups * 1024 +
    selected_input_bytes
  let layout = cuda_projection_operand_layout(
    width,
    padded=padded_bytes <= 49152,
  )
  let stride = layout.row_stride()
  let transport : CudaProjectionTransportLayout = { columns: width }
  let slot = if selected && !legacy_selected {
    (rows + columns) * width
  } else {
    layout.storage_elements(rows + columns)
  }
  let input_elements = if selected && !legacy_selected {
    rows * width
  } else {
    layout.storage_elements(rows)
  }
  let logical_elements = (rows + columns) * width
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
  let fragment_declarations = if legacy_selected {
    (
      $|      using namespace nvcuda;
      $|      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      $|      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
      $|      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[\{fold_rows}];
      $|      #pragma unroll
      $|      for (int row = 0; row < \{fold_rows}; ++row) wmma::fill_fragment(c[row], 0.0f);
    )
  } else {
    (
      $|      float c[\{fold_rows}][8] = {};
    )
  }
  let fragment_fold = if legacy_selected {
    (
      $|            wmma::load_matrix_sync(b, current + \{input_elements} + consumer_column * \{layout.offset(16, 0)} + k, \{stride});
      $|            #pragma unroll
      $|            for (int row = 0; row < \{fold_rows}; ++row) {
      $|              if (row_begin + (consumer_row + row) * 16 >= pipeline_rows) break;
      $|              wmma::load_matrix_sync(a, current + (consumer_row + row) * \{layout.offset(16, 0)} + k, \{stride});
      $|              wmma::mma_sync(c[row], a, b, c[row]);
      $|            }
    )
  } else if selected {
    (
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
    )
  } else {
    (
      $|            unsigned b[4];
      $|            #pragma unroll
      $|            for (int part = 0; part < 4; ++part)
      $|              b[part] = *reinterpret_cast<const unsigned *>(current + \{input_elements} + operand_index(consumer_column * 16 + lane / 4 + (part & 1) * 8, k + (lane % 4) * 2 + (part / 2) * 8));
      $|            #pragma unroll
      $|            for (int row = 0; row < \{fold_rows}; ++row) {
      $|              if (row_begin + (consumer_row + row) * 16 >= pipeline_rows) break;
      $|              unsigned a[4];
      $|              #pragma unroll
      $|              for (int part = 0; part < 4; ++part)
      $|                a[part] = *reinterpret_cast<const unsigned *>(current + operand_index((consumer_row + row) * 16 + lane / 4 + (part & 1) * 8, k + (lane % 4) * 2 + (part / 2) * 8));
      $|              #pragma unroll
      $|              for (int half = 0; half < 2; ++half) {
      $|                float *d = c[row] + half * 4;
      $|                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half]), "r"(b[half+2]));
      $|              }
      $|            }
    )
  }
  let fragment_store = if legacy_selected {
    (
      $|          wmma::store_matrix_sync(temporary, c[row], 16, wmma::mem_row_major);
    )
  } else {
    (
      $|          #pragma unroll
      $|          for (int item = 0; item < 8; ++item)
      $|            temporary[result_index(lane / 4 + (item % 4 / 2) * 8, lane % 4 * 2 + item % 2 + item / 4 * 8)] = c[row][item];
    )
  }
  let source =
    $|  const int pipeline_rows = \{row_count};
    $|  if (pipeline_rows >= \{plan.minimum_rows()}\{if selected { " && pipeline_rows <= " + maximum_rows.to_string() } else { "" }}) {
    $|    __shared__ __align__(32) __nv_bfloat16 matrix_storage[\{slot * 2}];
    $|    extern __shared__ unsigned char matrix_output_bytes[];
    $|    float *temporary = reinterpret_cast<float *>(matrix_output_bytes) + warp * 256;
    $|    \{if legacy_selected { "" } else { transport.address_function() }}
    $|    const int column_groups = \{(output_width + columns - 1) / columns};
    $|    const int work_count = ((pipeline_rows + \{rows - 1}) / \{rows}) * column_groups;
    $|    for (int work = blockIdx.x; work < work_count; work += gridDim.x) {
    $|      const int row_begin = (work / column_groups) * \{rows};
    $|      const int column_begin = (work % column_groups) * \{columns};
    $|      const int consumer_row = (warp / \{distribution.output_groups()}) * \{fold_rows};
    $|      const int consumer_column = warp % \{distribution.output_groups()};
    $|      auto stage_operands = [&](int k_base, int stage) {
    $|        __nv_bfloat16 *destination = matrix_storage + stage * \{slot};
    $|        for (int vector = threadIdx.x; vector < \{logical_elements / 8}; vector += blockDim.x) {
    $|          const int element = vector * 8;
    $|          const bool is_input = element < \{rows * width};
    $|          const int offset = is_input ? element : element - \{rows * width};
    $|          const int local_row = offset / \{width}, k = offset % \{width};
    $|          const int logical_row = row_begin + local_row;
    $|          const int token_row = \{input_row};
    $|          const int logical_column = column_begin + local_row;
    $|          \{weight_selection}
    $|          const bool valid = is_input ? (logical_row < pipeline_rows && token_row >= 0 && token_row < token_count) : logical_column < \{output_width};
    $|          const __nv_bfloat16 *source = input;
    $|          if (valid) source = is_input ? input + (unsigned long long)token_row * \{input_width}ULL + k_base + k : base + (unsigned long long)weight_row * \{input_width}ULL + k_base + k;
    $|          const int physical_element = (is_input ? 0 : \{input_elements}) + \{if legacy_selected { "local_row * " + stride.to_string() + " + k" } else { "operand_index(local_row, k)" }};
    $|          const unsigned address = (unsigned)__cvta_generic_to_shared(destination + physical_element);
    $|          const int bytes = valid ? 16 : 0;
    $|          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(bytes) : "memory");
    $|        }
    $|        asm volatile("cp.async.commit_group;" ::: "memory");
    $|      };
    $|\{fragment_declarations}
    $|      stage_operands(0, 0);
    $|      asm volatile("cp.async.wait_group 0;" ::: "memory");
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{input_width}; inner += \{width}) {
    $|        if (inner + \{width} < \{input_width}) stage_operands(inner + \{width}, stage ^ 1);
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
    $|\{fragment_store}
    $|          __syncwarp();
    $|          #pragma unroll
    $|          for (int item = 0; item < 8; ++item) {
    $|            const int element = lane + item * 32;
    $|            const int logical_row = row_begin + (consumer_row + row) * 16 + element / 16;
    $|            const int token_row = \{input_row};
    $|            const int column = column_begin + consumer_column * 16 + element % 16;
    $|            if (logical_row < pipeline_rows && token_row >= 0 && token_row < token_count && column < \{output_width}) output[(unsigned long long)token_row * \{output_width}ULL + column] = __float2bfloat16_rn(temporary[\{if legacy_selected { "element" } else { "result_index(element / 16, element % 16)" }}]);
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
