///|
test "export compressed index selection numerical probe kernels" {
  if @env.get_env_var("LUNA_EXPORT_COMPRESSED_SELECTION") != Some("1") {
    return
  }
  println("LUNA_COMPRESSED_SELECTION_SOURCE_BEGIN")
  for context in [1040, 3] {
    let plan = @precision.CompressedIndexSelectionPrecision::new(
      head_count=2,
      head_width=128,
      compression_ratio=4,
      maximum_rows=3,
      maximum_positions=context,
      top_k=8,
      maximum_workspace_bytes=4096L,
    )
    println(
      @utf8.decode(
        @compressed_index_selection_cuda_source.compressed_index_selection_source(
          plan,
          symbol_prefix=if context == 3 { "selection0" } else { "selection260" },
        ).source_bytes(),
      ),
    )
  }
  println("LUNA_COMPRESSED_SELECTION_SOURCE_END")
}
