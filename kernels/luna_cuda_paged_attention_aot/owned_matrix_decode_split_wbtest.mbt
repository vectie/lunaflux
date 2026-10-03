///|
test "new owned and matrix laws propagate through split AOT with exact geometry" {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=8,
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(9),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=16384,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  for
    id in [
      454, 455, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 480,
      481, 482, 483,
    ] {
    let compiled = compile_blockwise_decode_split(
      shape,
      profile,
      target,
      16384L,
      16384L,
      4096,
      8,
      16777216L,
      candidate_id=id,
    )
    let lowering = compiled.lowering()
    assert_eq(lowering.key_value_partitions(), 8)
    assert_eq(
      lowering.launch(),
      blockwise_decode_split_launch_geometry(shape, 8, candidate_id=id),
    )
    assert_eq(
      lowering.launch(),
      blockwise_decode_execution_launch_geometry(
        shape,
        8,
        compiled.functional().plan().candidate().compute(),
        lowering.tile().1,
        lowering.pipeline().0,
      ),
    )
    assert_eq(lowering.tile().1, if id % 2 == 0 { 32 } else { 64 })
    assert_eq(compiled.source().workspace().total_bytes(), 2129920L)
    let source = @utf8.decode(compiled.source().bytes())
    let law = match lowering.physical_program() {
      BlockwiseKeyProgram(fold, effects) => {
        assert_eq(effects.independently_refills(), id >= 460)
        fold.numeric_law()
      }
      GroupedMatrixKeyProgram(fold, effects) => {
        assert_eq(effects.independently_refills(), id >= 482)
        fold.numeric_law()
      }
      _ => fail("split adapter lost physical compute family")
    }
    assert_true(source.contains("numeric_law=\{law}"))
    assert_true(source.contains("decode_split_partial_v3_ep_3903"))
    assert_true(source.contains("decode_split_merge_v3_ep_3904"))
    if id == 460 || id == 461 {
      assert_eq(law, "dual-score-blockwise-f32-probability-v2")
    }
    assert_eq(lowering.launch().2, if id >= 480 { 32 } else { 64 })
  }
  try blockwise_decode_split_launch_geometry(shape, 8, candidate_id=420) catch {
    _ => ()
  } noraise {
    _ => fail("non-blockwise compiler family admitted by split adapter")
  }
}
