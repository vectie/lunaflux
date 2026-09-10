///|
/// Terminal lowering of one mapped matrix fold. Row selection and operand
/// lifetime come from the functional schedule; the same fragment operation is
/// used for directly gathered, streamed and resident input storage.
fn projection_matrix_map_source(
  parameters : ProjectionParameters,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> String {
  if compilation.selected_row_gather() is Some(gather) {
    return projection_selected_row_gather_source(parameters, gather)
  }
  let schedule = compilation.schedule()
  let selected = schedule.demand() == QueryRowEnds
  let resident = schedule.selected_input_row_resident()
  let width = if resident { parameters.input_width } else { 16 }
  let groups = schedule.output_tiles_per_workgroup()
  let layout : CudaProjectionTransportLayout = { columns: width }
  let gather = if selected {
    (
      $|    for (int element = threadIdx.x; element < \{16 * width}; element += blockDim.x) {
      $|      const int local_row = element / \{width}, component = element % \{width};
      $|      const int query_row = row + local_row;
      $|      const int token_row = query_row < row_count ? row_offsets[query_row + 1] - 1 : -1;
      $|      selected_input[operand_index(local_row, component)] = token_row >= 0 && token_row < token_count ? input[(unsigned long long)token_row * \{parameters.input_width}ULL + strip_begin + component] : __float2bfloat16_rn(0.0f);
      $|    }
      $|    __syncthreads();
    )
  } else {
    ""
  }
  let input_load = if selected {
    "a[part] = *reinterpret_cast<const unsigned *>(selected_input + operand_index(local_row, inner + local_k));"
  } else {
    "a[part] = row + local_row < token_count ? *reinterpret_cast<const unsigned *>(input + (unsigned long long)(row + local_row) * \{parameters.input_width}ULL + strip_begin + inner + local_k) : 0U;"
  }
  let weight_select = if parameters.family == QkvProjection {
    "const __nv_bfloat16 *base = query_weight; int weight_row = column + local_row; if (weight_row >= \{parameters.query_width}) { weight_row -= \{parameters.query_width}; if (weight_row >= \{parameters.intermediate_width}) { weight_row -= \{parameters.intermediate_width}; base = value_weight; } else base = key_weight; }"
  } else {
    "const __nv_bfloat16 *base = weight; const int weight_row = column + local_row;"
  }
  let input_storage = if selected {
    "__nv_bfloat16 *selected_input = reinterpret_cast<__nv_bfloat16 *>(map_bytes + \{groups * 1024});"
  } else {
    ""
  }
  let row_start = if selected { "0" } else { "row_begin" }
  let row_end = if selected { "row_count" } else { "row_begin + 1" }
  (
    $|  extern __shared__ unsigned char map_bytes[];
    $|  float *temporary = reinterpret_cast<float *>(map_bytes) + warp * 256;
    $|  \{input_storage}
    $|  \{layout.address_function()}
    $|  \{projection_result_address_source()}
    $|  for (int row = \{row_start}; row < \{row_end}; row += 16) {
    $|    float accumulator[8] = {};
    $|    for (int strip_begin = 0; strip_begin < \{parameters.input_width}; strip_begin += \{width}) {
    $|\{gather}
    $|      #pragma unroll
    $|      for (int inner = 0; inner < \{width}; inner += 16) {
    $|        unsigned a[4], b[4];
    $|        #pragma unroll
    $|        for (int part = 0; part < 4; ++part) {
    $|          const int local_row = lane / 4 + (part & 1) * 8;
    $|          const int local_k = (lane % 4) * 2 + (part / 2) * 8;
    $|          \{input_load}
    $|          \{weight_select}
    $|          b[part] = column + local_row < \{parameters.output_width} ? *reinterpret_cast<const unsigned *>(base + (unsigned long long)weight_row * \{parameters.input_width}ULL + strip_begin + inner + local_k) : 0U;
    $|        }
    $|        #pragma unroll
    $|        for (int half = 0; half < 2; ++half) {
    $|          float *d = accumulator + half * 4;
    $|          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half]), "r"(b[half+2]));
    $|        }
    $|      }
    $|      \{if selected { "__syncthreads();" } else { "" }}
    $|    }
    $|    #pragma unroll
    $|    for (int item = 0; item < 8; ++item) temporary[result_index(lane / 4 + (item % 4 / 2) * 8, lane % 4 * 2 + item % 2 + item / 4 * 8)] = accumulator[item];
    $|    __syncwarp();
    $|    for (int element = lane; element < 256; element += 32) {
    $|      const int logical_row = row + element / 16;
    $|      const int token_row = \{if selected { "logical_row < row_count ? row_offsets[logical_row + 1] - 1 : -1" } else { "logical_row" }};
    $|      if (token_row >= 0 && token_row < token_count && column + element % 16 < \{parameters.output_width}) output[(unsigned long long)token_row * \{parameters.output_width}ULL + column + element % 16] = __float2bfloat16_rn(temporary[result_index(element / 16, element % 16)]);
    $|    }
    $|    __syncwarp();
    $|  }
  )
}
