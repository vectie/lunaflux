///|
async test "export chunk delta actual sources and exact launch metadata" {
  if @env.get_env_var("LUNAFLUX_CHUNK_DELTA_HEADER") is Some(path) {
    let out = StringBuilder()
    let cases = [
      (2, 4, 6, 4, 2),
      (4, 32, 16, 8, 4),
      (64, 128, 128, 16, 2),
      (64, 128, 128, 32, 1),
      (64, 128, 128, 1, 1),
    ]
    for index, (heads, keys, values, rows, sequences) in cases {
      let sources = [false, true].map(chunk => {
        let p = @precision.RecurrentDeltaPrecision::new(
          head_count=heads,
          key_dimension=keys,
          value_dimension=values,
          maximum_rows=rows,
          maximum_sequences=sequences,
          cache_slots=4,
          normalization_epsilon=0.000001,
          log_decay_lower_bound=-5.0,
          maximum_state_bytes=16777216L,
          evaluation=if chunk {
            ChunkTriangularFloat32
          } else {
            OrderedTokenFloat32
          },
        )
        @recurrent_delta_execution_cuda_source.recurrent_delta_source(
          p,
          symbol=(if chunk { "chunk" } else { "ordered" }) + index.to_string(),
        )
      })
      for source in sources {
        out.write_string(@utf8.decode(source.source_bytes()))
        out <+ "\n"
      }
      out <+
        "void launch\{index}(bool chunk,int*c,int*o,int*s,int*r,__nv_bfloat16*q,__nv_bfloat16*k,__nv_bfloat16*v,float*g,__nv_bfloat16*b,float*h,__nv_bfloat16*y,float*w) {\n"
      for route, source in sources {
        out.write_string(
          if route == 0 {
            "if(!chunk) {\n"
          } else {
            "} else {\n"
          },
        )
        for stage in 0..<source.stage_count() {
          let (x, y, z, t) = source.stage_launch(stage)
          let extra = if route == 0 { "" } else { ",w" }
          out <+
            "\{source.stage_symbol(stage)}<<<dim3(\{x},\{y},\{z}),\{t}>>>(c,o,s,r,q,k,v,g,b,h,y\{extra});\n"
        }
      }
      out <+ "} }\n"
      out <+
        "constexpr size_t scratch\{index}=\{sources[1].workspace_bytes()/4L}ULL;\n"
    }
    @fs.write_file(path, out.to_string(), create_mode=CreateNew)
  }
}

///|
test "chunk arithmetic is explicit and its frame launches propagate" {
  let p = @precision.RecurrentDeltaPrecision::new(
    head_count=2,
    key_dimension=4,
    value_dimension=6,
    maximum_rows=4,
    maximum_sequences=2,
    cache_slots=4,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_state_bytes=768L,
    evaluation=ChunkTriangularFloat32,
  )
  let s = @recurrent_delta_execution_cuda_source.recurrent_delta_source(
    p,
    symbol="causal",
  )
  assert_eq(s.stage_count(), 3)
  assert_eq(s.stage_launch(0), (2, 2, 1, 128))
  assert_eq(s.stage_launch(1), (2, 2, 1, 32))
  assert_eq(s.stage_launch(2), (2, 2, 1, 128))
  assert_eq(s.workspace_bytes(), 3072L)
  let text = @utf8.decode(s.source_bytes())
  assert_true(text.contains("causal_prepare"))
  assert_true(text.contains("causal_solve"))
  assert_true(text.contains("causal_publish"))
  assert_true(text.contains("local_inverse[i*5+column]=x"))
  assert_true(!text.contains("__expf"))
}
