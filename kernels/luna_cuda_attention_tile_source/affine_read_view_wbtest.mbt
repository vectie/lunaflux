///|
test "affine producer guarantee reaches query-owned terminal lowering" {
  for id in [318, 320, 322, 323, 324] {
    let generic = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(id),
      dense_current=true,
    )
    let affine = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(id),
      affine_current=true,
    )
    assert_true(affine.reads_dense_current_key_value())
    assert_true(affine.affine_current_positions())
    assert_false(generic.affine_current_positions())
    assert_false(affine.digest() == generic.digest())
    let emit = lowering => {
      @utf8.decode(
        emit_cuda_attention_tile_source(
          lowering,
          source_test_abi(),
          "affine_view",
          1048576,
        ).bytes(),
      )
    }
    let ordinary = emit(generic)
    let optimized = emit(affine)
    assert_true(ordinary.contains("positions[begin+current_delta]==position"))
    assert_false(optimized.contains("positions[begin+current_delta]==position"))
    // Bounds, paging, invalid-page handling and zero-fill remain necessary.
    assert_true(
      optimized.contains("current_delta>=0 && current_delta<end-begin"),
    )
    assert_true(optimized.contains("page_indices[page_begin+position/"))
    assert_true(optimized.contains("atomicExch(control,0)"))
    assert_true(
      optimized.contains("const int bytes=kv_source_valid?16:0") || id == 318,
    )
    assert_true(optimized.contains("const float value=expf(delta)"))
    assert_eq(emit(affine), optimized)
  }
}

///|
test "export compiler affine prefill comparison" {
  if @env.get_env_var("LUNA_EXPORT_AFFINE_PREFILL") != Some("1") {
    return
  }
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=16384L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=16384,
  )
  let abi = { ..abi, query_metadata: true, }
  for affine in [false, true] {
    let lowering = repair_prefill_lifetime_lowering(322, affine_current=affine)
    let source = emit_cuda_attention_tile_source(
      lowering, abi, "lunaflux_attention_prefill_tile_compiler_v1", 1048576,
    )
    println("LUNA_SOURCE_BEGIN compiler-affine-\{affine}")
    println(@utf8.decode(source.bytes()))
    println("LUNA_SOURCE_END")
  }
}
