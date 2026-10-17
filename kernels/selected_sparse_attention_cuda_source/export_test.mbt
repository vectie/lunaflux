///|
/// Offline probe exports the exact selected lowering plus frozen serial leaf.
/// Neither benchmark inputs nor comparison enter the numerical/physical plans.
async test "export selected sparse paired probe" {
  if @env.get_env_var("LUNAFLUX_SPARSE_PROBE_HEADER") is Some(path) {
    let prefix = @env.get_env_var("LUNA_SPARSE_PREFIX_PROBE") == Some("1")
    let cases = if prefix {
      [
        (22, 32, 32, 256, 256, 2051, 64, 22),
        (1, 32, 32, 256, 256, 2051, 64, 64),
        (1, 2, 1, 256, 256, 2051, 4096, 2048),
        (3, 6, 2, 17, 33, 11, 32, 7),
      ]
    } else {
      [
        (1, 64, 64, 256, 256, 2051, 64, 16),
        (1, 64, 64, 256, 256, 2051, 64, 32),
        (1, 64, 64, 256, 256, 2051, 64, 64),
        (1, 2, 1, 256, 256, 2051, 4096, 2048),
        (3, 6, 2, 17, 33, 11, 32, 7),
        (1, 2, 1, 8, 8, 65536, 32, 16),
      ]
    }
    let code = StringBuilder()
    if prefix {
      code.write_string("#define LUNA_PREFIX_PROBE 1\n")
    }
    let metadata = StringBuilder()
    for i, (q, h, kv, d, v, w, t, live) in cases {
      let old = if prefix {
        @selected_sparse_attention_cuda_source.selected_sparse_source(
          query_heads=h,
          kv_heads=kv,
          qk_dimension=d,
          value_dimension=v,
          selected_width=w,
          maximum_queries=q,
          maximum_history=t,
          symbol="old_\{i}",
        ).source_bytes()
      } else {
        @serial.selected_index_sparse_attention_bf16_source(
          query_head_count=h,
          key_value_head_count=kv,
          query_key_dimension=d,
          value_dimension=v,
          selected_width=w,
          maximum_tokens=t,
          maximum_queries=q,
          function_symbol="old_\{i}",
        ).source_bytes()
      }
      let fresh = @selected_sparse_attention_cuda_source.selected_sparse_source(
        query_heads=h,
        kv_heads=kv,
        qk_dimension=d,
        value_dimension=v,
        selected_width=w,
        maximum_queries=q,
        maximum_history=t,
        symbol="new_\{i}",
        published_slots?=if prefix {
          Some(Int::min(w, t / 4 * 4 + 3))
        } else {
          None
        },
      )
      code.write_string(@utf8.decode(old))
      code.write_string(@utf8.decode(fresh.source_bytes()))
      let (grid, block) = fresh.launch()
      metadata.write_string(
        "{\{q},\{h},\{kv},\{d},\{v},\{w},\{t},\{live},\{grid},\{block},\{fresh.scratch_bytes()}ULL,(const void*)old_\{i},(const void*)new_\{i}},\n",
      )
    }
    code.write_string(
      "\nstatic Case cases[] = {\n" + metadata.to_string() + "};\n",
    )
    @fs.write_file(path, code.to_string(), create_mode=CreateNew)
  }
}
