///|
test "shape-aware dense MLP propagates plan and physical launches" {
  for
    (law, name) in [
      (@precision.DenseDotLaw::ScalarFloat32, "dense_old"),
      (ShapeAwareFloat32, "dense_new"),
    ] {
    let plan = @precision.DenseSwiGluPrecision::new(
      hidden_width=64,
      intermediate_width=96,
      maximum_rows=17,
      dot_law=law,
    )
    let source = @dense_mlp_execution_cuda_source.dense_mlp_source(
      plan,
      symbol_prefix=name,
    )
    let text = @utf8.decode(source.source_bytes())
    if law == ShapeAwareFloat32 {
      assert_eq(source.launch(0), (24, 128))
      assert_true(text.contains("if(live==1)"))
      assert_true(text.contains("mma_sync"))
      assert_true(text.contains("__shfl_down_sync"))
    } else {
      assert_eq(source.launch(0), (7, 256))
    }
    if @env.get_env_var("LUNA_EXPORT_SHAPE_AWARE") == Some("1") {
      println("LUNA_SOURCE_BEGIN")
      println(text)
      for stage in 0..<2 {
        let (g, b) = source.launch(stage)
        println(
          "static constexpr int \{name}_grid\{stage}=\{g},\{name}_block\{stage}=\{b};",
        )
      }
      println("LUNA_SOURCE_END")
    }
  }
}
