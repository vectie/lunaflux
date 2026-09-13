///|
test "offline observations select a candidate and reject stale or malformed tables" {
  let backend = @strategy.AttentionBackendId::new("test-backend")
  let target = @strategy.AttentionTargetId::new("test-target")
  let problem = @strategy.AttentionCompileProblem::new(
    shape=@strategy.AttentionShape::new(
      query_heads=8,
      kv_heads=2,
      head_dimension=64,
      tokens_per_page=16,
    ),
    phase=Prefill,
    numeric_type=AttentionBFloat16,
    query_rows=1024,
    context_tokens=8192,
    causal=true,
    allow_alternative_softmax_reduction=true,
    key_value_read_view=DenseCurrentAndPagedHistory,
    required_fragment_ownership=Some(QueryOwnedFragments),
  )
  let capabilities = @strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=100000L,
    max_pipeline_stages=2,
    supports_async_copy=true,
    matrix_shape=Some(
      @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
    ),
  )
  let candidates = @strategy.generate_attention_tile_candidates(
    problem, capabilities,
  )
  let header = "luna-attention-tuning-v1\nscope\tfrontier-device-toolchain\n"
  let workloads = "workload\t1024\t1\t0\nworkload\t2048\t8\t128\n"
  let valid = header +
    workloads +
    "record\t318\t200000\t7\nrecord\t322\t100000\t7\n"
  let records = parse_attention_tuning_records(
    @utf8.encode(valid),
    scope="frontier-device-toolchain",
    backend~,
    target~,
    problem~,
    candidates[:],
  )
  assert_eq(records.length(), 2)
  let selected = @strategy.compile_attention_tile_plan(
    records[:],
    backend~,
    target~,
    problem~,
    capabilities~,
  )
  assert_eq(selected.candidate().stable_id(), 322)
  for
    invalid in [
      valid.replace(old="frontier-device-toolchain", new="stale"),
      header + "record\t318\t1\t3\n",
      header + workloads + "record\t999\t1\t3\n",
      header + workloads + "record\t318\t1\t3\nrecord\t318\t2\t3\n",
      header + workloads + "record\t318\t1\t2\n",
      header + workloads + "record\t318\t0\t3\n",
      header + "workload\t1\t8\t0\nrecord\t318\t1\t3\n",
      header + workloads,
    ] {
    try
      parse_attention_tuning_records(
        @utf8.encode(invalid),
        scope="frontier-device-toolchain",
        backend~,
        target~,
        problem~,
        candidates[:],
      )
    catch {
      InvalidTable => ()
    } noraise {
      _ => fail("invalid table accepted")
    }
  }
}
