///|
async test "export fused HC control and Sinkhorn tail" {
  if @env.get_env_var("LUNA_HC_FUSION_HEADER") is Some(path) {
    let text = StringBuilder()
    let previous = @env.get_env_var("LUNA_HC_TAIL_BASELINE_HEADER").filter(p => {
      !p.is_empty()
    })
    if previous is Some(previous) {
      let mut old = @fs.read_file(previous).text()
      for i in 0..<4 {
        old = old
          .replace_all(old="tail\{i}", new="oldtail\{i}")
          .replace_all(old="launch\{i}", new="oldlaunch\{i}")
      }
      text.write_string(old)
      text <+ "\n"
    }
    for
      index, (streams, hidden, rows) in [
        (4, 64, 3),
        (4, 320, 4),
        (4, 4096, 64),
        (16, 64, 2),
      ] {
      let precision = @precision.HyperConnectionPrecision::new(
        stream_count=streams,
        hidden_width=hidden,
        maximum_rows=rows,
        sinkhorn_iterations=20,
        control_epsilon=1.0e-6,
        normalization_epsilon=1.0e-5,
        mix_law=RoundedBf16ControlsAndProducts,
        control_square_law=VectorizedTreeFloat32(partitions=256, vector_width=1),
        control_dot_law=StripedTreeFloat32(partitions=256),
      )
      let source = hyper_connection_source(
        precision,
        symbol_prefix="tail\{index}",
      )
      text.write_string(@utf8.decode(source.source_bytes()))
      text <+
        "\nvoid launch\{index}(bool fused,int *counts,__nv_bfloat16 *x,float *scratch,float *base,float *scale,float *post,float *initial,float *combine,__nv_bfloat16 *collapsed,__nv_bfloat16 *weight,__nv_bfloat16 *normalized) {\n"
      text <+
        "if(fused) \{source.symbol(7)}<<<\{rows},256>>>(counts,x,scratch,base,scale,post,initial,collapsed,weight,normalized,combine); else "
      text.write_string(
        if previous is Some(_) {
          "{oldtail\{index}_tail_function_tree_tiles_v1<<<\{rows},256>>>(counts,x,scratch,base,scale,post,initial,collapsed,weight,normalized,combine);}}\n"
        } else {
          "{\{source.symbol(0)}<<<\{rows},256>>>(counts,x,scratch,base,scale,post,initial,collapsed,weight,normalized); \{source.symbol(1)}<<<\{rows},1>>>(counts,initial,combine);}}\n"
        },
      )
    }
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}

///|
test "parallel initial controls are selected only in a legal fused subgroup" {
  let make = streams => {
    @precision.HyperConnectionPrecision::new(
      stream_count=streams,
      hidden_width=4096,
      maximum_rows=64,
      sinkhorn_iterations=20,
      control_epsilon=1.0e-6,
      normalization_epsilon=1.0e-5,
      mix_law=RoundedBf16ControlsAndProducts,
      control_square_law=VectorizedTreeFloat32(partitions=256, vector_width=1),
      control_dot_law=StripedTreeFloat32(partitions=256),
    )
  }
  let four = @utf8.decode(
    hyper_connection_source(make(4), symbol_prefix="four").source_bytes(),
  )
  let large = @utf8.decode(
    hyper_connection_source(make(16), symbol_prefix="large").source_bytes(),
  )
  assert_true(four.contains("const bool all_valid=__all_sync"))
  assert_false(large.contains("const bool all_valid=__all_sync"))
  assert_true(four.contains("if (worker == 0)")) // ordinary pre remains available
  assert_true(four.contains("sum=__fadd_rn(sum,__shfl_sync"))
}
