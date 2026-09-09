///|
/// The same sibling dot product and F32 SiLU expression are either retained
/// in the planned local arena or recomputed inside the ordered output fold.
/// Storage selection never changes intermediate rounding or the launch ABI.
fn projection_scalar_mlp_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  materialize : Bool,
) -> Bytes {
  let siblings = projection_dot_fold_source(
      parameters.input_width,
      [
        {
          accumulator: "gate",
          left: "__bfloat162float(input[input_base + inner])",
          right: "__bfloat162float(gate_weight[projection_base + inner])",
        },
        {
          accumulator: "up",
          left: "__bfloat162float(input[input_base + inner])",
          right: "__bfloat162float(up_weight[projection_base + inner])",
        },
      ],
      OrderedDot,
    ) +
    "const float silu = gate / __fadd_rn(1.0f, expf(-gate));\n" +
    "const float product = __fmul_rn(silu, up);\n"
  let body = if materialize {
    (
      $|  const int token = blockIdx.x;
      $|  if (token >= token_count) return;
      $|  const unsigned long long input_base = (unsigned long long)token * \{parameters.input_width}ULL;
      $|  extern __shared__ float combined[];
      $|  for (int intermediate = threadIdx.x; intermediate < \{parameters.intermediate_width}; intermediate += blockDim.x) {
      $|    const unsigned long long projection_base = (unsigned long long)intermediate * \{parameters.input_width}ULL;
      $|\{siblings}
      $|    combined[intermediate] = product;
      $|  }
      $|  __syncthreads();
      $|  for (int output_column = threadIdx.x; output_column < \{parameters.output_width}; output_column += blockDim.x) {
      $|    const unsigned long long index = (unsigned long long)token * \{parameters.output_width}ULL + output_column;
    )
  } else {
    (
      $|  const unsigned long long index = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (index >= (unsigned long long)token_count * \{parameters.output_width}ULL) return;
      $|  const int output_column = index % \{parameters.output_width}ULL;
      $|  const unsigned long long input_base = (index / \{parameters.output_width}ULL) * \{parameters.input_width}ULL;
    )
  }
  let value = if materialize {
    "const float product = combined[intermediate];\n"
  } else {
    "const unsigned long long projection_base = (unsigned long long)intermediate * \{parameters.input_width}ULL;\n" +
    siblings
  }
  @utf8.encode(
    projection_cuda_prelude() +
    (
      $|extern "C" __global__ void \{symbol}(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, const __nv_bfloat16 *down_weight, __nv_bfloat16 *output) {
      $|\{projection_count_guard(maximum_tokens)}
      $|\{body}
      $|  const unsigned long long down_base = (unsigned long long)output_column * \{parameters.intermediate_width}ULL;
      $|  float output_accumulator = 0.0f;
      $|  for (int intermediate = 0; intermediate < \{parameters.intermediate_width}; ++intermediate) {
      $|\{value}
      $|    output_accumulator = __fadd_rn(output_accumulator, __fmul_rn(product, __bfloat162float(down_weight[down_base + intermediate])));
      $|  }
      $|  output[index] = __float2bfloat16_rn(output_accumulator);
      $|\{if materialize { "  }" } else { "" }}
      $|}
    ),
  )
}
