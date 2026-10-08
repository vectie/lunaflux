///|
fn joint_decode_request() -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest raise {
  @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
    backend=@luna_attention_strategy.AttentionBackendId::new("cuda"),
    target_id=@luna_attention_strategy.AttentionTargetId::new(
      "sm121-joint-test",
    ),
    problem=@luna_attention_strategy.AttentionCompileProblem::new(
      shape=@luna_attention_strategy.AttentionShape::new(
        query_heads=16,
        kv_heads=4,
        head_dimension=128,
        tokens_per_page=16,
      ),
      phase=Decode,
      numeric_type=AttentionBFloat16,
      query_rows=1,
      batch_rows=1,
      context_tokens=4096,
      causal=true,
      allow_alternative_softmax_reduction=true,
    ),
    capabilities=@luna_attention_strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=49152L,
      max_pipeline_stages=2,
      supports_async_copy=true,
      matrix_shape=Some(
        @luna_attention_strategy.AttentionMatrixShape::new(
          rows=16,
          columns=16,
          reduction=16,
        ),
      ),
    ),
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    maximum_shared_memory_bytes=49152L,
    abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
      query_heads=16,
      key_value_heads=4,
      head_dimension=128,
      tokens_per_page=16,
      input_row_width=3072,
      page_stride_values=8192L,
      physical_page_capacity=1024L,
      maximum_query_rows=8,
      maximum_query_tokens=8,
      maximum_page_entries=2048,
    ),
    partial_symbol="joint_partial",
    merge_symbol="joint_merge",
    maximum_workspace_bytes=16777216L,
    maximum_source_bytes=1048576,
  )
}

///|
test "joint decode emits complete SIMT and matrix chains with distinct symbols" {
  let request = joint_decode_request()
  let frontier = @luna_cuda_attention_tile_aot.enumerate_cuda_partitioned_attention_tiles(
    request,
    [452, 480, 482],
    [8, 16, 32],
  )
  assert_eq(frontier.variants().length(), 9)
  assert_eq(
    frontier,
    @luna_cuda_attention_tile_aot.enumerate_cuda_partitioned_attention_tiles(
      request,
      [482, 452, 480],
      [32, 8, 16],
    ),
  )
  let symbols : Array[String] = []
  for variant in frontier.variants() {
    let compiled = variant.compilation()
    let (partial_view, merge_view) = compiled.source().symbols()
    let partial = partial_view.to_owned()
    let merge = merge_view.to_owned()
    assert_false(symbols.contains(partial))
    assert_false(symbols.contains(merge))
    symbols.append([partial, merge])
    let source = @utf8.decode(compiled.source().bytes())
    assert_true(source.contains("void " + partial + "("))
    assert_true(source.contains("void " + merge + "("))
    assert_true(compiled.source().workspace().total_bytes() > 0L)
    assert_true(
      compiled.functional().schedule().key_value_partitions() is (2 | 4 | 8),
    )
  }
}

///|
test "whole chain selection retains scratch tradeoffs and returns exact AOT artifact" {
  let frontier = @luna_cuda_attention_tile_aot.enumerate_cuda_partitioned_attention_tiles(
    joint_decode_request(),
    [452, 480],
    [8, 32],
  )
  let scope = "device\ttoolchain-flags\tC1/H4096\tlaunch-context"
  let observations = frontier
    .variants()
    .mapi((index, variant) => {
      @luna_cuda_attention_tile_aot.PartitionedAttentionChainObservation::new(
        frontier=frontier.digest(),
        compilation=variant.compilation().digest(),
        scope~,
        elapsed_ns=1000L - index.to_int64() * 100L,
        samples=7,
      )
    })
  let selected = frontier.select_measured_chain(
    scope~,
    observations[:],
    maximum_workspace_bytes=16777216L,
    maximum_shared_memory_bytes=49152L,
  )
  assert_eq(selected, frontier.variants()[3].compilation())
  let small = frontier.variants()[0]
    .compilation()
    .source()
    .workspace()
    .total_bytes()
  let constrained = frontier.select_measured_chain(
    scope~,
    observations[:],
    maximum_workspace_bytes=small,
    maximum_shared_memory_bytes=49152L,
  )
  assert_eq(constrained, frontier.variants()[1].compilation())
  assert_eq(
    frontier.select_measured_chain(
      scope~,
      observations.rev()[:],
      maximum_workspace_bytes=small,
      maximum_shared_memory_bytes=49152L,
    ),
    constrained,
  )
  for limit in [0L, small - 1L] {
    try
      frontier.select_measured_chain(
        scope~,
        observations[:],
        maximum_workspace_bytes=limit,
        maximum_shared_memory_bytes=49152L,
      )
    catch {
      @luna_cuda_attention_tile_aot.InvalidCudaAttentionTileAot(InvalidRequest) =>
        ()
      _ => fail("unexpected failure")
    } noraise {
      _ => fail("unmeasured or infeasible fallback")
    }
  }
}

///|
test "whole chain selection rejects stale scope duplicate records and weak samples" {
  let request = joint_decode_request()
  let frontier = @luna_cuda_attention_tile_aot.enumerate_cuda_partitioned_attention_tiles(
    request,
    [452],
    [8, 32],
  )
  let other = @luna_cuda_attention_tile_aot.enumerate_cuda_partitioned_attention_tiles(
    request,
    [480],
    [8, 32],
  )
  let observation = @luna_cuda_attention_tile_aot.PartitionedAttentionChainObservation::new(
    frontier=frontier.digest(),
    compilation=frontier.variants()[0].compilation().digest(),
    scope="correct",
    elapsed_ns=100L,
    samples=3,
  )
  let foreign = @luna_cuda_attention_tile_aot.PartitionedAttentionChainObservation::new(
    frontier=other.digest(),
    compilation=other.variants()[0].compilation().digest(),
    scope="correct",
    elapsed_ns=50L,
    samples=3,
  )
  for
    (scope, records) in [
      ("wrong", [observation]),
      ("correct", [observation, observation]),
      ("correct", [foreign]),
      ("correct", []),
    ] {
    try
      frontier.select_measured_chain(
        scope~,
        records,
        maximum_workspace_bytes=16777216L,
        maximum_shared_memory_bytes=49152L,
      )
    catch {
      @luna_cuda_attention_tile_aot.InvalidCudaAttentionTileAot(InvalidRequest) =>
        ()
      _ => fail("unexpected failure")
    } noraise {
      _ => fail("invalid chain comparison accepted")
    }
  }
  try
    @luna_cuda_attention_tile_aot.PartitionedAttentionChainObservation::new(
      frontier=frontier.digest(),
      compilation=frontier.variants()[0].compilation().digest(),
      scope="correct",
      elapsed_ns=100L,
      samples=2,
    )
  catch {
    @luna_cuda_attention_tile_aot.InvalidCudaAttentionTileAot(InvalidRequest) =>
      ()
    _ => fail("unexpected failure")
  } noraise {
    _ => fail("insufficient samples accepted")
  }
}
