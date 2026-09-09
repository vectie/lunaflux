///|
fn projection_cuda_prelude() -> String {
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
fn projection_tensor_core_prelude() -> String {
  "#include <cuda_bf16.h>\n#include <mma.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
fn projection_tensor_core_eligible(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> Bool {
  parameters.family != GatedMlp &&
  dispatch.prefill().kernel_class() == MatrixTile
}

///|
fn projection_tensor_core_mlp_eligible(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> Bool {
  parameters.family == GatedMlp &&
  dispatch.prefill().kernel_class() == MatrixTile
}

///|
fn projection_decode_workspace_bytes(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> Int64 {
  if projection_tensor_core_mlp_eligible(parameters, dispatch) {
    parameters.intermediate_width.to_int64() * 2L
  } else {
    0L
  }
}

///|
fn projection_workspace_bytes(
  parameters : ProjectionParameters,
  profile : @launch_contract.PagedKernelProfile,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
) -> Int64 raise ProjectionAotError {
  let row_bytes = projection_decode_workspace_bytes(parameters, dispatch)
  if row_bytes == 0L {
    0L
  } else {
    projection_checked_mul(row_bytes, profile.max_query_tokens().to_int64())
  }
}

///|
fn projection_count_guard(maximum_tokens : Int) -> String {
  "  const int token_count = counts[3];\n" +
  "  if (token_count <= 0 || token_count > " +
  maximum_tokens.to_string() +
  ") return;\n"
}

///|
fn projection_tensor_core_direct_dense_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> Bytes {
  let demand : @luna_projection_tile_compiler.ProjectionResultDemand = tile_compilation
    .schedule()
    .demand()
  let query_row_ends = demand == QueryRowEnds
  let groups = tile_compilation.schedule().output_tiles_per_workgroup()
  let cooperative_column_tail = query_row_ends &&
    parameters.output_width % (groups * 16) != 0
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    (if query_row_ends {
      "(const int32_t *counts, const int32_t *row_offsets, const __nv_bfloat16 *input, "
    } else {
      "(const int32_t *counts, const __nv_bfloat16 *input, "
    }) +
    "const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    projection_matrix_pipeline_source(
      parameters, tile_compilation, maximum_rows,
    ) +
    (if !query_row_ends {
      "  if (token_count == 1) {\n" +
      "    const int output_column = (int)blockIdx.x * \{groups} + warp; if (output_column >= " +
      parameters.output_width.to_string() +
      ") return;\n" +
      "    const unsigned long long weight_base = (unsigned long long)output_column * " +
      parameters.input_width.to_string() +
      "ULL;\n" +
      projection_dot_fold_source(
        parameters.input_width,
        [
          {
            accumulator: "value",
            left: "__bfloat162float(input[inner])",
            right: "__bfloat162float(weight[weight_base + inner])",
          },
        ],
        tile_compilation.schedule().single_row_reduction(),
      ) +
      "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
      "    return;\n  }\n" +
      "  const int group_count = " +
      ((parameters.output_width + groups * 16 - 1) / (groups * 16)).to_string() +
      ";\n" +
      "  const int group_index = (int)blockIdx.x % group_count;\n" +
      "  const int row_begin = ((int)blockIdx.x / group_count) * 16;\n" +
      "  const int column = (group_index * \{groups} + warp) * 16;\n" +
      "  if (row_begin >= token_count || column >= " +
      parameters.output_width.to_string() +
      ") return;\n"
    } else {
      "  const int column = ((int)blockIdx.x * \{groups} + warp) * 16;\n" +
      (if cooperative_column_tail {
        ""
      } else {
        "  if (column >= " + parameters.output_width.to_string() + ") return;\n"
      })
    }) +
    (if query_row_ends {
      "  const int row_count = counts[2];\n" +
      "  if (row_count <= 0 || row_count > " +
      maximum_rows.to_string() +
      ") return;\n" +
      "  if (counts[2] == 1) {\n" +
      "    const int selected_row = row_offsets[1] - 1;\n" +
      "    if (selected_row < 0 || selected_row >= token_count) return;\n" +
      "    const unsigned long long live_row = (unsigned long long)selected_row;\n" +
      "    const unsigned long long input_base = live_row * " +
      parameters.input_width.to_string() +
      "ULL;\n" +
      "    const unsigned long long output_base = live_row * " +
      parameters.output_width.to_string() +
      "ULL;\n" +
      "    #pragma unroll\n" +
      "    for (int item = 0; item < 16; ++item) {\n" +
      "      const int output_column = column + item; if (output_column >= " +
      parameters.output_width.to_string() +
      ") break;\n" +
      "      const unsigned long long weight_base = (unsigned long long)output_column * " +
      parameters.input_width.to_string() +
      "ULL;\n" +
      projection_dot_fold_source(
        parameters.input_width,
        [
          {
            accumulator: "value",
            left: "__bfloat162float(input[input_base + inner])",
            right: "__bfloat162float(weight[weight_base + inner])",
          },
        ],
        tile_compilation.schedule().single_row_reduction(),
      ) +
      "      if (lane == 0) output[output_base + output_column] = __float2bfloat16_rn(value);\n" +
      "    } return;\n" +
      "  }\n"
    } else {
      ""
    }) +
    (if tile_compilation.schedule().matrix_pipeline() is Some(_) {
      ""
    } else {
      projection_matrix_map_source(parameters, tile_compilation)
    }) +
    "}\n",
  )
}

///|
fn projection_tensor_core_direct_qkv_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  groups? : Int = 8,
) -> Bytes {
  @utf8.encode(
    projection_tensor_core_prelude() +
    "extern \"C\" __global__ void " +
    symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, " +
    "const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, " +
    "const __nv_bfloat16 *value_weight, __nv_bfloat16 *output) {\n" +
    projection_count_guard(maximum_tokens) +
    "  const int warp = (int)threadIdx.x >> 5;\n" +
    "  const int lane = (int)threadIdx.x & 31;\n" +
    projection_matrix_pipeline_source(
      parameters, tile_compilation, maximum_tokens,
    ) +
    "  if (token_count == 1) {\n" +
    "    const int output_column = (int)blockIdx.x * \{groups} + warp; if (output_column >= " +
    parameters.output_width.to_string() +
    ") return;\n" +
    "    const __nv_bfloat16 *selected_weight = query_weight;\n" +
    "    int selected_column = output_column;\n" +
    "    if (output_column >= " +
    parameters.query_width.to_string() +
    ") { selected_column -= " +
    parameters.query_width.to_string() +
    "; if (selected_column >= " +
    parameters.intermediate_width.to_string() +
    ") { selected_column -= " +
    parameters.intermediate_width.to_string() +
    "; selected_weight = value_weight; } else selected_weight = key_weight; }\n" +
    "    const unsigned long long weight_base = (unsigned long long)selected_column * " +
    parameters.input_width.to_string() +
    "ULL;\n" +
    projection_dot_fold_source(
      parameters.input_width,
      [
        {
          accumulator: "value",
          left: "__bfloat162float(input[inner])",
          right: "__bfloat162float(selected_weight[weight_base + inner])",
        },
      ],
      tile_compilation.schedule().single_row_reduction(),
    ) +
    "    if (lane == 0) output[output_column] = __float2bfloat16_rn(value);\n" +
    "    return;\n  }\n" +
    (if tile_compilation.schedule().matrix_pipeline() is Some(_) {
      ""
    } else {
      "  const int group_count = \{(parameters.output_width + groups * 16 - 1) / (groups * 16)};\n" +
      "  const int row_begin = ((int)blockIdx.x / group_count) * 16;\n" +
      "  const int column = ((int)blockIdx.x % group_count * \{groups} + warp) * 16;\n" +
      "  if (row_begin >= token_count || column >= \{parameters.output_width}) return;\n" +
      projection_matrix_map_source(parameters, tile_compilation)
    }) +
    "}\n",
  )
}
