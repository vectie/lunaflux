///|
test "reusable partition binding consumes typed KV64 storage rather than a candidate ID" {
  let bytes = b"owned-kv64-partition-module"
  let binding = @luna_cuda_paged_attention_aot.bind_reusable_paged_attention_decode_split_cuda_aot(
    pa_plan().operations()[2],
    pa_decode_split_profile(1024),
    pa_layout(),
    pa_target(),
    pa_decode_split_operands(1024),
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    @catalog.AotKernelEntryPointId::from_stable_int(41),
    @catalog.AotKernelEntryPointId::from_stable_int(42),
    8,
    1048576L,
    blockwise_readonly=true,
    blockwise_execution=(
      @luna_attention_strategy.AttentionGroupedQueryOwnedBlockwiseDot(
        score_owners=4,
        contraction=true,
      ),
      64,
      2,
    ),
  )
  let partial = binding.contract().launches()[0]
  assert_eq(partial.dimensions().block_x(), 64)
  assert_eq(partial.dimensions().shared_memory_bytes(), 1552)
  assert_eq(partial.dimensions().grid_z(), 8)
  assert_eq(
    partial.function_symbol(),
    "lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_41",
  )
}
