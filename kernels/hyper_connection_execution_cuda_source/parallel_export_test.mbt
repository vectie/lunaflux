///|
async test "export ordered hyper control tiles paired probe" {
  if @env.get_env_var("LUNAFLUX_HYPER_TILES_HEADER") is Some(path) {
    let text = StringBuilder()
    let cases = [(2, 17, 3), (4, 129, 4), (4, 4096, 16), (16, 7, 2)]
    for index, (streams, hidden, rows) in cases {
      let p = @precision.HyperConnectionPrecision::new(
        stream_count=streams,
        hidden_width=hidden,
        maximum_rows=rows,
        sinkhorn_iterations=20,
        control_epsilon=1.0e-6,
        normalization_epsilon=1.0e-5,
        mix_law=RoundedBf16ControlsAndProducts,
      )
      let old = hyper_connection_source(
        p,
        symbol_prefix="old\{index}",
        parallel_controls=false,
      )
      let new = hyper_connection_source(p, symbol_prefix="new\{index}")
      let schedule = new.control_schedule().unwrap()
      assert_eq(
        schedule.workspace_bytes(),
        rows.to_int64() * (streams * (streams + 2) + 1).to_int64() * 4L,
      )
      assert_eq(new.symbol(4), "new\{index}_square_ordered_tiles_v1")
      text.write_string(@utf8.decode(old.source_bytes()))
      text <+ "\n"
      text.write_string(@utf8.decode(new.source_bytes()))
      text <+
        "\nvoid launch\{index}(bool parallel, int *counts, __nv_bfloat16 *x, __nv_bfloat16 *w, float *base, float *scale, float *scratch, float *post, float *matrix, __nv_bfloat16 *y) {\n"
      text <+
        "if (parallel) { \{new.symbol(4)}<<<\{rows},256>>>(counts,x,scratch); \{new.symbol(5)}<<<\{schedule.workgroups(1)},256>>>(counts,x,w,scratch); \{new.symbol(0)}<<<\{rows},256>>>(counts,x,scratch,base,scale,post,matrix,y); } else { \{old.symbol(0)}<<<\{rows},1>>>(counts,x,w,base,scale,post,matrix,y); } }\n"
    }
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}
