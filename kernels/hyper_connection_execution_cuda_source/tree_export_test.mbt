///|
async test "export parallel hyper tree and rounded fusion probe" {
  if @env.get_env_var("LUNAFLUX_HYPER_TREE_HEADER") is Some(path) {
    let text = StringBuilder()
    for
      index, (streams, hidden, rows) in [
        (4, 64, 3),
        (4, 320, 4),
        (4, 4096, 16),
        (16, 64, 2),
      ] {
      let p = @precision.HyperConnectionPrecision::new(
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
      let old = hyper_connection_source(
        p,
        symbol_prefix="old\{index}",
        parallel_controls=false,
      )
      let new = hyper_connection_source(p, symbol_prefix="new\{index}")
      assert_true(new.fuses_normalization())
      text.write_string(@utf8.decode(old.source_bytes()))
      text.write_string(@utf8.decode(new.source_bytes()))
      text <+
        "\nvoid launch\{index}(bool parallel, int *counts, __nv_bfloat16 *x, __nv_bfloat16 *w, float *base, float *scale, float *scratch, float *post, float *matrix, __nv_bfloat16 *y, __nv_bfloat16 *norm_weight, __nv_bfloat16 *normalized) {\n"
      text <+
        "if (parallel) { \{new.symbol(4)}<<<\{rows},256>>>(counts,x,scratch); \{new.symbol(5)}<<<\{rows * streams * (streams+2)},256>>>(counts,x,w,scratch); \{new.symbol(0)}<<<\{rows},256>>>(counts,x,scratch,base,scale,post,matrix,y,norm_weight,normalized); } else { \{old.symbol(0)}<<<\{rows},1>>>(counts,x,w,base,scale,post,matrix,y); \{old.symbol(2)}<<<\{rows},256>>>(counts,y,norm_weight,normalized); } }\n"
    }
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}

///|
test "tree fusion is permitted only for identical norm ownership" {
  let p = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=64,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
    control_square_law=VectorizedTreeFloat32(partitions=128, vector_width=1),
    control_dot_law=StripedTreeFloat32(partitions=128),
  )
  let source = hyper_connection_source(p, symbol_prefix="tree128")
  assert_false(source.fuses_normalization())
  assert_eq(source.control_schedule().unwrap().workers(), 128)
  assert_eq(source.symbol(5), "tree128_project_tree_tiles_v1")
}
