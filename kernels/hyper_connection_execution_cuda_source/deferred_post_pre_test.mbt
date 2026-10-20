///|
async test "export deferred HC post pre alternatives" {
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
      text.write_string(@utf8.decode(source.source_bytes()))
      for group in [1, 2, 4] {
        text <+ "\n"
        text.write_string(
          @utf8.decode(
            @control.deferred_post_pre_source(
              precision,
              precision,
              controls_per_group=group,
              symbol="deferred\{index}_\{group}",
            ),
          ),
        )
      }
      let buckets = [(1, 1)]
      if rows > 1 {
        buckets.push((Int::min(rows, 32), 2))
      }
      if rows > 32 {
        buckets.push((rows, 4))
      }
      let dispatch = @physical.DeferredPostPreDispatch::new(
        precision,
        precision,
        buckets,
        workgroup_budget=128,
      )
      text <+ "\n"
      text.write_string(
        @utf8.decode(
          @control.deferred_post_pre_dispatch_source(
            precision,
            precision,
            dispatch,
            symbol="dispatch\{index}",
          ),
        ),
      )
      let wider = @physical.DeferredPostPreDispatch::new(
        precision,
        precision,
        buckets.map(b => (b.0, if b.0 > 32 { 8 } else { b.1 })),
        workgroup_budget=128,
      )
      text <+ "\n"
      text.write_string(
        @utf8.decode(
          @control.deferred_post_pre_dispatch_source(
            precision,
            precision,
            wider,
            symbol="wide_dispatch\{index}",
          ),
        ),
      )
      text <+ "\n"
      text.write_string(
        @utf8.decode(
          @control.deferred_post_pre_dispatch_source(
            precision,
            precision,
            dispatch,
            symbol="shared_dispatch\{index}",
            subgroup_width=1,
          ),
        ),
      )
      text <+
        "\nvoid launch\{index}(int variant,int *counts,float *post,float *combine,__nv_bfloat16 *residual,__nv_bfloat16 *branch,__nv_bfloat16 *output,__nv_bfloat16 *weight,float *scratch,cudaStream_t stream) {\n"
      text <+
        "if(variant==8) dispatch\{index}<<<\{dispatch.workgroups()},256,0,stream>>>(counts,post,combine,residual,branch,output,weight,scratch);\n"
      text <+
        "if(variant==9) shared_dispatch\{index}<<<\{dispatch.workgroups()},256,0,stream>>>(counts,post,combine,residual,branch,output,weight,scratch);\n"
      text <+
        "if(variant==10) wide_dispatch\{index}<<<\{wider.workgroups()},256,0,stream>>>(counts,post,combine,residual,branch,output,weight,scratch);\n"
      text <+
        "if(variant==0) { \{source.symbol(3)}<<<\{(rows*streams*hidden+255)/256},256,0,stream>>>(counts,post,combine,residual,branch,output); \{source.symbol(4)}<<<\{rows},256,0,stream>>>(counts,output,scratch); \{source.symbol(5)}<<<\{rows*streams*(streams+2)},256,0,stream>>>(counts,output,weight,scratch); }\n"
      for group in [1, 2, 4] {
        let plan = @physical.DeferredPostPrePlan::new(
          precision,
          precision,
          controls_per_group=group,
          workgroup_budget=128,
        )
        text <+
          "if(variant==\{group}) deferred\{index}_\{group}<<<\{plan.workgroups()},256,0,stream>>>(counts,post,combine,residual,branch,output,weight,scratch);\n"
      }
      text <+ "}\n"
    }
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}
