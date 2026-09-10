///|
/// The materialized sibling product and its ordered consumer share the same
/// fold lowering at every admitted matrix extent, including masked row tails.
fn projection_matrix_mlp_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  max_registers~ : Int,
) -> Bytes {
  let schedule = compilation.schedule()
  let groups = schedule.output_tiles_per_workgroup()
  let down_groups = schedule.companion_output_tiles_per_workgroup().unwrap()
  let pipeline = schedule.intermediate_pipeline().unwrap()
  let sibling = schedule.sibling_pipeline().unwrap()
  let launch_bounds = projection_sibling_launch_bounds_source(
    schedule,
    max_registers~,
  )
  let gate_fold = projection_dot_fold_source(
    parameters.input_width,
    [
      {
        accumulator: "gate",
        left: "__bfloat162float(input[inner])",
        right: "__bfloat162float(gate_weight[weight_base + inner])",
      },
      {
        accumulator: "up",
        left: "__bfloat162float(input[inner])",
        right: "__bfloat162float(up_weight[weight_base + inner])",
      },
    ],
    schedule.single_row_reduction(),
  )
  let outputs = ["value", "second", "third", "fourth"]
  let dots : Array[CudaProjectionDot] = []
  let stores = StringBuilder()
  for index, name in outputs {
    dots.push({
      accumulator: name,
      left: "__bfloat162float(workspace[inner])",
      right: "__bfloat162float(down_weight[weight_base + \{index * parameters.intermediate_width}ULL + inner])",
    })
    stores <+
      "if (lane == 0) output[output_column + \{index}] = __float2bfloat16_rn(\{name});\n"
  }
  let down_fold = projection_dot_fold_source(
    parameters.intermediate_width,
    dots[:],
    schedule.single_row_reduction(),
  )
  let matrix_fold = projection_fold_pipeline_source(
    pipeline,
    "workspace + (unsigned long long)row_start * \{parameters.intermediate_width}ULL",
    "down_weight + (unsigned long long)(group_index * \{down_groups * 16}) * \{parameters.intermediate_width}ULL",
    parameters.intermediate_width,
    down_groups,
    parameters.output_width,
  )
  @utf8.encode(
    projection_tensor_core_prelude() +
    (
      $|extern "C" __global__ \{launch_bounds}void \{symbol}(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output, __nv_bfloat16 *workspace) {
      $|\{projection_count_guard(maximum_tokens)}
      $|  const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
      $|  if (blockDim.x != \{groups * 32}) return;
      $|  if (token_count == 1) {
      $|    const int intermediate = (int)blockIdx.x * \{groups} + warp;
      $|    if (intermediate >= \{parameters.intermediate_width}) return;
      $|    const unsigned long long weight_base = (unsigned long long)intermediate * \{parameters.input_width}ULL;
      $|\{gate_fold}
      $|    if (lane == 0) workspace[intermediate] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
      $|    return;
      $|  }
      $|\{projection_sibling_fold_pipeline_source(sibling, parameters, groups~)}
      $|}
      $|extern "C" __global__ void \{symbol}_down(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output, __nv_bfloat16 *workspace) {
      $|\{projection_count_guard(maximum_tokens)}
      $|  if (blockDim.x != \{down_groups * 32}) return;
      $|  const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
      $|  if (token_count == 1) {
      $|    const int output_column = ((int)blockIdx.x * \{down_groups} + warp) * 4;
      $|    if (output_column >= \{parameters.output_width}) return;
      $|    const unsigned long long weight_base = (unsigned long long)output_column * \{parameters.intermediate_width}ULL;
      $|\{down_fold}
      $|\{stores.to_string()}
      $|    return;
      $|  }
      $|  const int group_count = \{parameters.output_width / (down_groups * 16)};
      $|  const int live_groups = ((token_count + \{pipeline.row_width() - 1}) / \{pipeline.row_width()}) * group_count;
      $|  extern __shared__ unsigned char shared_bytes[];
      $|  float *warp_output = reinterpret_cast<float *>(shared_bytes) + warp * 256;
      $|  for (int work_group = (int)blockIdx.x; work_group < live_groups; work_group += (int)gridDim.x) {
      $|    const int group_index = work_group % group_count;
      $|    const int row_start = (work_group / group_count) * \{pipeline.row_width()};
      $|    const int column = group_index * \{down_groups * 16} + warp * 16;
      $|\{matrix_fold}
      $|  }
      $|}
    ),
  )
}
