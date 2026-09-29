///|
fn postprocess_effect_source(
  program : @elementwise_physical_ir.IngressProgram,
) -> String {
  let source = StringBuilder()
  for effect in program.effects() {
    source.write_string(
      match effect {
        LoadHead =>
          "    for (int component = lane; component < LF_HEAD_DIM; component += LF_LANES_PER_HEAD)\n" +
          "      head_values[component] = __bfloat162float(projected_qkv[head_base + component]);\n"
        PublishHead | RetireHead => "    __syncwarp();\n"
        NormalizeRotateQueryKey =>
          "    if (head < LF_QK_HEADS) {\n" +
          "      const __nv_bfloat16* norm_weight =\n" +
          "        head < LF_QUERY_HEADS ? query_norm_weight : key_norm_weight;\n" +
          "      lf_ingress_norm_rotary(head_values, norm_weight, position, lane);\n" +
          "    }\n"
        StoreActivationAndCache =>
          "    for (int component = lane; component < LF_HEAD_DIM; component += LF_LANES_PER_HEAD) {\n" +
          "      float value = head_values[component];\n" +
          "      const __nv_bfloat16 encoded = __float2bfloat16_rn(value);\n" +
          "      rotated_qkv[head_base + component] = encoded;\n" +
          "      if (head >= LF_QUERY_HEADS) {\n" +
          "        const bool key = head < LF_QK_HEADS;\n" +
          "        const int kv_head = key ? head - LF_QUERY_HEADS : head - LF_QK_HEADS;\n" +
          "        const unsigned long long cache_index =\n" +
          "          (unsigned long long)physical_page * LF_PAGE_STRIDE_BF16 +\n" +
          "          (unsigned long long)(position % LF_TOKENS_PER_PAGE) * LF_KV_WIDTH +\n" +
          "          (unsigned long long)kv_head * LF_HEAD_DIM + (unsigned long long)component;\n" +
          "        (key ? key_cache : value_cache)[cache_index] = encoded;\n" +
          "      }\n" +
          "    }\n"
      },
    )
  }
  source.to_string()
}
