///|
priv struct CudaProjectionDot {
  accumulator : String
  left : String
  right : String
}

///|
/// Lower a product of pure scalar folds. Reassociation is explicitly supplied
/// by the numeric schedule; sharing a loop does not grant reassociation.
fn projection_dot_fold_source(
  extent : Int,
  dots : ArrayView[CudaProjectionDot],
  reduction : @luna_projection_tile_compiler.ProjectionSingleRowReduction,
) -> String {
  let builder = StringBuilder()
  for dot in dots {
    builder <+ "float \{dot.accumulator} = 0.0f;\n"
  }
  let (start, stride) = match reduction {
    OrderedDot => ("0", 1)
    StridedPairwiseDot(lanes) => ("lane", lanes)
  }
  builder <+
    "for (int inner = \{start}; inner < \{extent}; inner += \{stride}) {\n"
  for dot in dots {
    builder <+
      "  \{dot.accumulator} = __fadd_rn(\{dot.accumulator}, __fmul_rn(\{dot.left}, \{dot.right}));\n"
  }
  builder <+ "}\n"
  if reduction is StridedPairwiseDot(lanes) {
    builder <+ "for (int delta = \{lanes / 2}; delta > 0; delta >>= 1) {\n"
    for dot in dots {
      builder <+
        "  \{dot.accumulator} = __fadd_rn(\{dot.accumulator}, __shfl_down_sync(0xffffffffU, \{dot.accumulator}, delta));\n"
    }
    builder <+ "}\n"
  }
  builder.to_string()
}

///|
fn projection_scalar_map_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> Bytes {
  let selected = compilation.schedule().demand() == QueryRowEnds
  let segmented = parameters.family == QkvProjection
  let weights = if segmented {
    "const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, const __nv_bfloat16 *value_weight"
  } else {
    "const __nv_bfloat16 *weight"
  }
  let select_weight = if segmented {
    "const __nv_bfloat16 *weight = query_weight; int local_column = column; if (local_column >= \{parameters.query_width}) { local_column -= \{parameters.query_width}; if (local_column >= \{parameters.intermediate_width}) { local_column -= \{parameters.intermediate_width}; weight = value_weight; } else weight = key_weight; }"
  } else {
    "const int local_column = column;"
  }
  let fold = projection_dot_fold_source(
    parameters.input_width,
    [
      {
        accumulator: "accumulator",
        left: "__bfloat162float(input[input_base + inner])",
        right: "__bfloat162float(weight[weight_base + inner])",
      },
    ],
    OrderedDot,
  )
  @utf8.encode(
    projection_cuda_prelude() +
    (
      $|extern "C" __global__ void \{symbol}(const int32_t *counts, \{if selected { "const int32_t *row_offsets, " } else { "" }}const __nv_bfloat16 *input, \{weights}, __nv_bfloat16 *output) {
      $|\{projection_count_guard(maximum_tokens)}
      $|  const int row_count = \{if selected { "counts[2]" } else { "token_count" }};
      $|  if (row_count <= 0 || row_count > \{if selected { maximum_rows } else { maximum_tokens }}) return;
      $|  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  const unsigned long long live = (unsigned long long)row_count * \{parameters.output_width}ULL;
      $|  if (index >= live) return;
      $|  const int logical_row = index / \{parameters.output_width}ULL;
      $|  const int row = \{if selected { "row_offsets[logical_row + 1] - 1" } else { "logical_row" }};
      $|  if (row < 0 || row >= token_count) return;
      $|  const int column = index % \{parameters.output_width}ULL;
      $|  \{select_weight}
      $|  const unsigned long long input_base = (unsigned long long)row * \{parameters.input_width}ULL;
      $|  const unsigned long long weight_base = (unsigned long long)local_column * \{parameters.input_width}ULL;
      $|\{fold}
      $|  output[index] = __float2bfloat16_rn(accumulator);
      $|}
    ),
  )
}
