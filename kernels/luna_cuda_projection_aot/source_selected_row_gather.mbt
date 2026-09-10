///|
/// Lower the single-consumer selected-row gather into register fragments.
/// The lane/part map and ordered m16n8k16 calls are identical to the shared
/// matrix map. Row addresses are invariant across K, and every result pair
/// has one owner, so neither operand nor result requires a shared roundtrip.
fn projection_selected_row_gather_source(
  parameters : ProjectionParameters,
  gather : @luna_projection_tile_compiler.ProjectionSelectedRowGather,
) -> String {
  let (rows, _, _) = gather.dimensions()
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
    $|\{projection_selected_row_prefetch_fold_source(parameters, gather)}
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

///|
/// Advance a bounded immutable operand window, consuming its members in the
/// original reduction order. Slot indices are resolved by MoonBit, not by
/// runtime indexing into a device array. The final partial window issues no
/// out-of-domain load and executes no extra zero-product reduction.
fn projection_selected_row_prefetch_fold_source(
  parameters : ProjectionParameters,
  gather : @luna_projection_tile_compiler.ProjectionSelectedRowGather,
) -> String {
  let (_, _, reduction) = gather.dimensions()
  let depth = gather.prefetch_fragments()
  let advance = depth * reduction
  let builder = StringBuilder()
  for slot in 0..<depth {
    builder <+ "    unsigned operand\{slot}_a[4], operand\{slot}_b[4];\n"
    builder.write_string(
      projection_selected_row_fragment_load_source(
        "operand\{slot}",
        "\{slot * reduction} + (lane % 4) * 2",
      ),
    )
  }
  builder <+ "    #pragma unroll 1\n"
  builder <+
    "    for (int inner = 0; inner < \{parameters.input_width}; inner += \{advance}) {\n"
  for slot in 0..<depth {
    let offset = slot * reduction
    builder <+ "      if (inner + \{offset} < \{parameters.input_width}) {\n"
    builder <+ "        unsigned next_a[4] = {}, next_b[4] = {};\n"
    builder <+
      "        if (inner + \{advance + offset} < \{parameters.input_width}) {\n"
    builder.write_string(
      projection_selected_row_fragment_load_source(
        "next",
        "inner + \{advance + offset} + (lane % 4) * 2",
      ),
    )
    builder <+ "        }\n"
    builder.write_string(
      projection_selected_row_fragment_fold_source("operand\{slot}"),
    )
    for part in 0..<4 {
      builder <+ "        operand\{slot}_a[\{part}] = next_a[\{part}];\n"
      builder <+ "        operand\{slot}_b[\{part}] = next_b[\{part}];\n"
    }
    builder <+ "      }\n"
  }
  builder <+ "    }\n"
  builder.to_string()
}

///|
fn projection_selected_row_fragment_load_source(
  prefix : String,
  component : String,
) -> String {
  (
    $|      {
    $|        const int k = \{component};
    $|        \{prefix}_a[0] = valid_low ? *reinterpret_cast<const unsigned *>(input + input_low + k) : 0U;
    $|        \{prefix}_a[1] = valid_high ? *reinterpret_cast<const unsigned *>(input + input_high + k) : 0U;
    $|        \{prefix}_a[2] = valid_low ? *reinterpret_cast<const unsigned *>(input + input_low + k + 8) : 0U;
    $|        \{prefix}_a[3] = valid_high ? *reinterpret_cast<const unsigned *>(input + input_high + k + 8) : 0U;
    $|        \{prefix}_b[0] = *reinterpret_cast<const unsigned *>(weight + weight_low + k);
    $|        \{prefix}_b[1] = *reinterpret_cast<const unsigned *>(weight + weight_high + k);
    $|        \{prefix}_b[2] = *reinterpret_cast<const unsigned *>(weight + weight_low + k + 8);
    $|        \{prefix}_b[3] = *reinterpret_cast<const unsigned *>(weight + weight_high + k + 8);
    $|      }
    $|
  )
}

///|
fn projection_selected_row_fragment_fold_source(prefix : String) -> String {
  (
    $|        #pragma unroll
    $|        for (int half = 0; half < 2; ++half) {
    $|          float *d = accumulator + half * 4;
    $|          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(\{prefix}_a[0]), "r"(\{prefix}_a[1]), "r"(\{prefix}_a[2]), "r"(\{prefix}_a[3]), "r"(\{prefix}_b[half]), "r"(\{prefix}_b[half+2]));
    $|        }
    $|
  )
}
