///|
async test "export adjacent HC publication square fusion" {
  if @env.get_env_var("LUNA_HC_FUSION_HEADER") is Some(path) {
    let text = StringBuilder()
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
        symbol_prefix="hc\{index}",
      )
      assert_true(source.supports_post_square())
      text.write_string(@utf8.decode(source.source_bytes()))
      text <+
        "\nvoid launch\{index}(bool fused,int *counts,float *post,float *combine,__nv_bfloat16 *residual,__nv_bfloat16 *branch,__nv_bfloat16 *output,float *scratch) {\n"
      text <+
        "if(fused) \{source.symbol(6)}<<<\{rows},256>>>(counts,post,combine,residual,branch,output,scratch); else { \{source.symbol(3)}<<<\{(rows*streams*hidden+255)/256},256>>>(counts,post,combine,residual,branch,output); \{source.symbol(4)}<<<\{rows},256>>>(counts,output,scratch); } }\n"
    }
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}
