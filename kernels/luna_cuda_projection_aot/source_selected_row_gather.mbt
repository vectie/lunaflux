///|
/// Lower the single-consumer selected-row gather into register fragments.
/// The lane/part map and ordered m16n8k16 calls are identical to the shared
/// matrix map. Row addresses are invariant across K, and every result pair
/// has one owner, so neither operand nor result requires a shared roundtrip.
fn projection_selected_row_gather_source(
  parameters : ProjectionParameters,
  gather : @luna_projection_tile_compiler.ProjectionSelectedRowGather,
) -> String {
  let (rows, _, reduction) = gather.dimensions()
  (
    $|  for (int row = 0; row < row_count; row += \{rows}) {
    $|    const int logical_low = row + lane / 4;
    $|    const int logical_high = logical_low + 8;
    $|    const int token_low = logical_low < row_count ? row_offsets[logical_low + 1] - 1 : -1;
    $|    const int token_high = logical_high < row_count ? row_offsets[logical_high + 1] - 1 : -1;
    $|    const bool valid_low = token_low >= 0 && token_low < token_count;
    $|    const bool valid_high = token_high >= 0 && token_high < token_count;
    $|    const unsigned long long input_low = valid_low ? (unsigned long long)token_low * \{parameters.input_width}ULL : 0ULL;
    $|    const unsigned long long input_high = valid_high ? (unsigned long long)token_high * \{parameters.input_width}ULL : 0ULL;
    $|    const unsigned long long weight_low = (unsigned long long)(column + lane / 4) * \{parameters.input_width}ULL;
    $|    const unsigned long long weight_high = weight_low + \{8 * parameters.input_width}ULL;
    $|    float accumulator[8] = {};
    $|    #pragma unroll 1
    $|    for (int inner = 0; inner < \{parameters.input_width}; inner += \{reduction}) {
    $|      const int k = inner + (lane % 4) * 2;
    $|      unsigned a[4], b[4];
    $|      a[0] = valid_low ? *reinterpret_cast<const unsigned *>(input + input_low + k) : 0U;
    $|      a[1] = valid_high ? *reinterpret_cast<const unsigned *>(input + input_high + k) : 0U;
    $|      a[2] = valid_low ? *reinterpret_cast<const unsigned *>(input + input_low + k + 8) : 0U;
    $|      a[3] = valid_high ? *reinterpret_cast<const unsigned *>(input + input_high + k + 8) : 0U;
    $|      b[0] = *reinterpret_cast<const unsigned *>(weight + weight_low + k);
    $|      b[1] = *reinterpret_cast<const unsigned *>(weight + weight_high + k);
    $|      b[2] = *reinterpret_cast<const unsigned *>(weight + weight_low + k + 8);
    $|      b[3] = *reinterpret_cast<const unsigned *>(weight + weight_high + k + 8);
    $|      #pragma unroll
    $|      for (int half = 0; half < 2; ++half) {
    $|        float *d = accumulator + half * 4;
    $|        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half]), "r"(b[half+2]));
    $|      }
    $|    }
    $|    #pragma unroll
    $|    for (int pair = 0; pair < 4; ++pair) {
    $|      const bool high = (pair & 1) != 0;
    $|      const int token_row = high ? token_high : token_low;
    $|      const int output_column = column + lane % 4 * 2 + (pair / 2) * 8;
    $|      if (high ? valid_high : valid_low) {
    $|        const __nv_bfloat16 low = __float2bfloat16_rn(accumulator[pair * 2]);
    $|        const __nv_bfloat16 high_value = __float2bfloat16_rn(accumulator[pair * 2 + 1]);
    $|        output[(unsigned long long)token_row * \{parameters.output_width}ULL + output_column] = low;
    $|        output[(unsigned long long)token_row * \{parameters.output_width}ULL + output_column + 1] = high_value;
    $|      }
    $|    }
    $|  }
  )
}
