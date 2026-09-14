///|
test "attention ingress cannot accept wrapped output extents" {
  for
    (queries, keys, dimension) in [
      (8, 1073741824, 128),
      (536870912, 536870916, 128),
    ] {
    let epilogue = ProjectionAttentionIngressEpilogue::new(
      query_heads=queries,
      kv_heads=keys,
      head_dimension=dimension,
      norm_epsilon=0.000001,
      rope_theta=10000.0,
    )
    try
      ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
        dispatch=projection_compiler_dispatch(Qkv),
        maximum_rows=32,
        maximum_tokens=256,
        epilogue~,
      )
    catch {
      InvalidProjectionTileCompilation(InvalidDemand) => ()
      _ => fail("unexpected ingress extent error")
    } noraise {
      _ => fail("wrapped head extent accepted")
    }
  }
}
