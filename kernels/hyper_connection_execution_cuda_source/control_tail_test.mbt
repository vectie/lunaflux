///|
async test "export fused HC control and Sinkhorn tail" {
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
        symbol_prefix="tail\{index}",
      )
      text.write_string(@utf8.decode(source.source_bytes()))
      text <+
        "\nvoid launch\{index}(bool fused,int *counts,__nv_bfloat16 *x,float *scratch,float *base,float *scale,float *post,float *initial,float *combine,__nv_bfloat16 *collapsed,__nv_bfloat16 *weight,__nv_bfloat16 *normalized) {\n"
      text <+
        "if(fused) \{source.symbol(7)}<<<\{rows},256>>>(counts,x,scratch,base,scale,post,initial,collapsed,weight,normalized,combine); else {\{source.symbol(0)}<<<\{rows},256>>>(counts,x,scratch,base,scale,post,initial,collapsed,weight,normalized); \{source.symbol(1)}<<<\{rows},1>>>(counts,initial,combine);}}\n"
    }
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}
