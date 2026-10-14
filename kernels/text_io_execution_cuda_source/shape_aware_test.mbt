///|
test "parallel head and greedy retain explicit publication and stable tie" {
  for
    (law, name) in [
      (@precision.DenseDotLaw::ScalarFloat32, "text_old"),
      (ShapeAwareFloat32, "text_new"),
    ] {
    let plan = @precision.TextIoPrecision::new(
      vocabulary_size=1024,
      hidden_width=256,
      stream_count=4,
      maximum_rows=17,
      maximum_output_rows=2,
      normalization_epsilon=1.0e-5,
      maximum_workspace_bytes=100000L,
      head_law=law,
    )
    let source = @text_io_execution_cuda_source.text_io_source(
      plan,
      symbol_prefix=name,
    )
    let text = @utf8.decode(source.source_bytes())
    if law == ShapeAwareFloat32 {
      assert_eq(source.head_launch(), (256, 128))
      assert_eq(source.greedy_threads(), 256)
      assert_true(text.contains("other<ids[lane]"))
      assert_true(text.contains("invalid[lane]|=invalid[lane+span]"))
    } else {
      assert_eq(source.greedy_threads(), 1)
    }
    if @env.get_env_var("LUNA_EXPORT_SHAPE_AWARE") == Some("1") {
      println("LUNA_SOURCE_BEGIN")
      println(text)
      println("LUNA_SOURCE_END")
    }
  }
}
